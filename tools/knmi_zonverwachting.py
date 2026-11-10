"""KNMI-zonverwachting (HARMONIE-AROME) omzetten naar data/zonverwachting.json.

Proef (10-11-2026): kan de KNMI-verwachting forecast.solar in de BMSmachine
vervangen? Dit script maakt het bestand; de kastjes gebruiken het nog niet.

Bron: KNMI-dataset harmonie_arome_cy43_p1 (v1.0, CC BY 4.0). Elke run is een
tar met per verwachtingsuur (+00 t/m +60) een GRIB1-bestand voor het hele
domein. Daaruit alleen de globale straling (GRAD, indicatorOfParameter 117,
J/m2, opgeteld vanaf het begin van de run). Het uurgemiddelde in W/m2 is
(A[u] - A[u-1]) / 3600. Daarna gemiddeld over vakken van 0,2 x 0,2 graad
boven Nederland.

De tar wordt tijdens het downloaden gelezen (streaming): niets staat in z'n
geheel in het geheugen of op schijf.

Gebruik:
    KNMI_API_KEY=... python tools/knmi_zonverwachting.py <uitvoer.json> \\
        [--vorige <url>] [--status <status.json>] [--min-uren-tussen 3]

Met --vorige: is de run in het vorige bestand minder dan --min-uren-tussen
oud ten opzichte van de nieuwste run, dan wordt niets gedownload en komt het
vorige bestand in de uitvoer (exitcode 3).

Exitcodes: 0 = nieuw bestand, 3 = vorige behouden, 1 = fout (met --vorige:
dan staat het vorige bestand, als dat er is, toch in de uitvoer).
De sleutel wordt nooit gelogd of geprint.
"""

import argparse
import datetime as dt
import json
import os
import re
import sys
import tarfile
import time
import urllib.error
import urllib.parse
import urllib.request

import numpy as np

DATASET = "harmonie_arome_cy43_p1"
BASIS = ("https://api.dataplatform.knmi.nl/open-data/v1/datasets/"
         f"{DATASET}/versions/1.0/files")
BRON_URL = "https://dataplatform.knmi.nl/dataset/harmonie-arome-cy43-p1-1-0"
USER_AGENT = "BanthumLabs-zonverwachting/1.0 (+https://banthumlabs.nl; info@banthumlabs.nl)"

PARAM_GRAD = 117           # globale straling, J/m2, opgeteld vanaf de start
# Vakken boven Europees Nederland (lat/lon van de linkeronderhoek, stap in graden).
RASTER = {"lat0": 50.7, "lon0": 3.3, "stap": 0.2, "nlat": 15, "nlon": 20}
MIN_UREN = 48              # minder verwachtingsuren = fout (vandaag + morgen)
MAX_UREN = 60
MAX_WM2 = 1300             # hoger uurgemiddelde = fout
MAX_DOWNLOAD = 8 * 1024**3  # bytes; daarboven stoppen
MAX_GROOTTE = 200 * 1024   # bytes JSON
WACHTTIJDEN = (10, 30, 90)
RUN_RE = re.compile(r"(\d{10})")                       # JJJJMMDDUU in de tar-naam
LID_RE = re.compile(r"HA43_N20_(\d{12})_(\d{3})(\d{2})_GB")


class Fout(Exception):
    pass


def open_url(url, sleutel=None):
    headers = {"User-Agent": USER_AGENT}
    if sleutel:
        headers["Authorization"] = sleutel
    for poging in range(len(WACHTTIJDEN) + 1):
        try:
            return urllib.request.urlopen(urllib.request.Request(url, headers=headers), timeout=120)
        except urllib.error.HTTPError as e:
            opnieuw = e.code == 429 or e.code >= 500
            melding = f"HTTP {e.code}"
        except (urllib.error.URLError, TimeoutError, ConnectionError) as e:
            opnieuw = True
            melding = f"netwerkfout ({type(e).__name__})"
        if not opnieuw or poging == len(WACHTTIJDEN):
            raise Fout(melding)
        print(f"{melding}, opnieuw over {WACHTTIJDEN[poging]} s", file=sys.stderr)
        time.sleep(WACHTTIJDEN[poging])


def haal_json(url, sleutel=None, max_bytes=1024 * 1024):
    with open_url(url, sleutel) as r:
        data = r.read(max_bytes + 1)
    if len(data) > max_bytes:
        raise Fout("antwoord groter dan verwacht")
    return json.loads(data)


def run_tijd(naam):
    m = RUN_RE.search(naam)
    if not m:
        raise Fout(f"geen runtijd in bestandsnaam {naam}")
    return dt.datetime.strptime(m.group(1), "%Y%m%d%H").replace(tzinfo=dt.timezone.utc)


def nieuwste_run(sleutel):
    lijst = haal_json(BASIS + "?orderBy=filename&sorting=desc&maxKeys=1", sleutel)
    bestanden = lijst.get("files") or []
    if not bestanden:
        raise Fout("geen bestanden in de KNMI-lijst")
    return bestanden[0]["filename"], bestanden[0].get("size")


class Teller:
    """Bestandsobject dat stopt boven MAX_DOWNLOAD."""

    def __init__(self, f):
        self.f, self.n = f, 0

    def read(self, n=-1):
        b = self.f.read(n)
        self.n += len(b)
        if self.n > MAX_DOWNLOAD:
            raise Fout(f"download groter dan {MAX_DOWNLOAD // 1024**2} MB")
        return b


def grib1_berichten(buf):
    """(param, niveautype, niveau, bytes) per GRIB1-bericht in buf."""
    i = 0
    while True:
        i = buf.find(b"GRIB", i)
        if i < 0 or i + 8 > len(buf):
            return
        lengte = int.from_bytes(buf[i + 4:i + 7], "big")
        editie = buf[i + 7]
        if editie != 1 or lengte < 8 or i + lengte > len(buf):
            i += 4
            continue
        pds = i + 8
        yield buf[pds + 8], buf[pds + 9], int.from_bytes(buf[pds + 10:pds + 12], "big"), buf[i:i + lengte]
        i += lengte


def lees_veld(bericht):
    """values, lats, lons (1-D) van één GRIB-bericht."""
    import eccodes  # alleen hier nodig: zo blijft de rest zonder eccodes te testen
    gid = eccodes.codes_new_from_message(bytes(bericht))
    try:
        waarden = eccodes.codes_get_values(gid)
        lats = eccodes.codes_get_array(gid, "latitudes")
        lons = eccodes.codes_get_array(gid, "longitudes")
    finally:
        eccodes.codes_release(gid)
    return np.asarray(waarden, float), np.asarray(lats, float), np.asarray(lons, float)


def vak_index(lats, lons):
    """Per rasterpunt het vaknummer (rij * nlon + kolom), -1 buiten het raster."""
    r = np.floor((lats - RASTER["lat0"]) / RASTER["stap"] + 1e-9).astype(int)
    k = np.floor((lons - RASTER["lon0"]) / RASTER["stap"] + 1e-9).astype(int)
    binnen = (r >= 0) & (r < RASTER["nlat"]) & (k >= 0) & (k < RASTER["nlon"])
    return np.where(binnen, r * RASTER["nlon"] + k, -1)


def vak_gemiddelden(waarden, index):
    n = RASTER["nlat"] * RASTER["nlon"]
    ok = (index >= 0) & np.isfinite(waarden)
    som = np.bincount(index[ok], weights=waarden[ok], minlength=n)
    aantal = np.bincount(index[ok], minlength=n)
    if (aantal == 0).any():
        raise Fout(f"{int((aantal == 0).sum())} vakken zonder rasterpunten")
    return som / aantal


def uurgemiddelden(opgeteld):
    """opgeteld: {uur: array J/m2 per vak} -> array [vak, uur] in W/m2 (uur 0 = run+0..+1)."""
    uren = sorted(opgeteld)
    if uren[0] != 0:
        raise Fout("verwachtingsuur +00 ontbreekt")
    laatste = 0
    while laatste + 1 in opgeteld and laatste + 1 <= MAX_UREN:
        laatste += 1
    if laatste < MIN_UREN:
        raise Fout(f"maar {laatste} aaneengesloten verwachtingsuren (min {MIN_UREN})")
    A = np.stack([opgeteld[u] for u in range(laatste + 1)], axis=1)  # [vak, uur]
    wm2 = np.diff(A, axis=1) / 3600.0
    if (wm2 < -5).any():
        raise Fout("opgetelde straling neemt af: geen optelling vanaf de start?")
    wm2 = np.clip(wm2, 0, None)
    if wm2.max() > MAX_WM2:
        raise Fout(f"uurgemiddelde {wm2.max():.0f} W/m2 boven {MAX_WM2}")
    return np.rint(wm2).astype(int)


def verwerk_tar(stroom):
    """Leest de tar als stroom; geeft ({uur: J/m2 per vak}, aantal leden)."""
    opgeteld, index, leden = {}, None, 0
    with tarfile.open(fileobj=stroom, mode="r|*") as tar:
        for lid in tar:
            if not lid.isfile():
                continue
            m = LID_RE.search(lid.name)
            if not m:
                continue
            leden += 1
            uur = int(m.group(2))
            if uur > MAX_UREN:
                continue
            buf = tar.extractfile(lid).read()
            gevonden = [b for p, t, niv, b in grib1_berichten(buf) if p == PARAM_GRAD and t == 105 and niv == 0]
            if len(gevonden) != 1:
                raise Fout(f"{len(gevonden)} GRAD-velden in {lid.name} (verwacht 1)")
            waarden, lats, lons = lees_veld(gevonden[0])
            if index is None:
                index = vak_index(lats, lons)
            opgeteld[uur] = vak_gemiddelden(waarden, index)
    if not opgeteld:
        raise Fout(f"geen bruikbare GRIB-bestanden in de tar ({leden} leden herkend)")
    return opgeteld, leden


def maak_json(naam, run, wm2, mb, duur):
    return {
        "bron": "KNMI HARMONIE-AROME Cy43 (P1)",
        "licentie": "CC BY 4.0",
        "bron_url": BRON_URL,
        "bestand": naam,
        "run": run.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "gemaakt": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "uitleg": ("Globale straling op een horizontaal vlak, W/m2, gemiddeld over het uur dat begint op "
                   "run + i uur (i = 0, 1, ...). Per vak van stap x stap graad; vak = rij * nlon + kolom, "
                   "rij 0 = lat0, kolom 0 = lon0."),
        "raster": RASTER,
        "uren": int(wm2.shape[1]),
        "download_mb": mb,
        "duur_s": duur,
        "waarden": wm2.tolist(),
    }


def schrijf(pad, tekst):
    os.makedirs(os.path.dirname(os.path.abspath(pad)), exist_ok=True)
    with open(pad, "w", encoding="utf-8") as f:
        f.write(tekst)


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("uitvoer")
    p.add_argument("--vorige", help="URL van het huidige zonverwachting.json")
    p.add_argument("--status", help="pad voor een klein statusbestand (ook bij een fout)")
    p.add_argument("--min-uren-tussen", type=float, default=3.0)
    args = p.parse_args()

    status = {"tijd": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")}
    vorige_tekst = None
    if args.vorige:
        try:
            with open_url(args.vorige) as r:
                vorige_tekst = r.read(MAX_GROOTTE * 2).decode("utf-8")
            json.loads(vorige_tekst)
        except (Fout, ValueError, UnicodeDecodeError) as e:
            print(f"vorige zonverwachting.json niet bruikbaar ({e})", file=sys.stderr)
            vorige_tekst = None

    code = 1
    try:
        sleutel = os.environ.get("KNMI_API_KEY", "").strip()
        if not sleutel:
            raise Fout("KNMI_API_KEY ontbreekt")
        naam, grootte = nieuwste_run(sleutel)
        run = run_tijd(naam)
        status.update(nieuwste=naam, nieuwste_mb=round(grootte / 1024**2) if grootte else None)
        if vorige_tekst:
            vorige_run = dt.datetime.strptime(json.loads(vorige_tekst)["run"], "%Y-%m-%dT%H:%M:%SZ") \
                .replace(tzinfo=dt.timezone.utc)
            if (run - vorige_run).total_seconds() < args.min_uren_tussen * 3600:
                schrijf(args.uitvoer, vorige_tekst)
                status.update(resultaat="vorige behouden", run=json.loads(vorige_tekst)["run"])
                print(f"vorige run {vorige_run:%Y-%m-%d %H}Z is recent genoeg (nieuwste {naam})")
                code = 3
                return code

        info = haal_json(f"{BASIS}/{urllib.parse.quote(naam)}/url", sleutel)
        download_url = info.get("temporaryDownloadUrl")
        if not download_url:
            raise Fout("geen download-URL van het KNMI")
        begin = time.monotonic()
        # Ondertekende tijdelijke URL: geen sleutel meesturen.
        with open_url(download_url) as r:
            stroom = Teller(r)
            opgeteld, leden = verwerk_tar(stroom)
        duur = round(time.monotonic() - begin)
        mb = round(stroom.n / 1024**2)
        wm2 = uurgemiddelden(opgeteld)
        tekst = json.dumps(maak_json(naam, run, wm2, mb, duur), separators=(",", ":"))
        if len(tekst) > MAX_GROOTTE:
            raise Fout(f"zonverwachting.json is {len(tekst)} bytes (max {MAX_GROOTTE})")
        schrijf(args.uitvoer, tekst)
        status.update(resultaat="nieuw", run=run.strftime("%Y-%m-%dT%H:%M:%SZ"), uren=int(wm2.shape[1]),
                      download_mb=mb, duur_s=duur, bytes=len(tekst), tar_leden=leden)
        print(f"zonverwachting.json: run {run:%Y-%m-%d %H}Z, {wm2.shape[1]} uur, {mb} MB in {duur} s, "
              f"{len(tekst)} bytes")
        code = 0
        return code
    except Fout as e:
        status.update(resultaat="fout", fout=str(e))
        print(f"FOUT: {e}", file=sys.stderr)
        if vorige_tekst:
            schrijf(args.uitvoer, vorige_tekst)
        return 1
    except Exception as e:  # onverwacht (tar, GRIB): vorige behouden, kort melden
        status.update(resultaat="fout", fout=f"{type(e).__name__}: {str(e)[:200]}")
        print(f"FOUT: {type(e).__name__}: {str(e)[:200]}", file=sys.stderr)
        if vorige_tekst:
            schrijf(args.uitvoer, vorige_tekst)
        return 1
    finally:
        if args.status:
            schrijf(args.status, json.dumps(status, ensure_ascii=False, separators=(",", ":")))


if __name__ == "__main__":
    sys.exit(main())
