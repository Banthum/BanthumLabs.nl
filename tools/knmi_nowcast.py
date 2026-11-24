"""KNMI-satelliet-nowcast van de zonnestraling omzetten naar data/zonnowcast.json.

Proef (24-11-2026, schaduw): de kastjes loggen dit bestand naast de
HARMONIE-verwachting, maar gebruiken het nog niet. Ontwerp en terugtest:
kennisbank, BMSmachine_ontwerp_satelliet_nowcast.md en §10.

Bron: KNMI-dataset surface_solar_irradiance (v1.0, CC BY 4.0). Elke kwartier
een run: één GRIB2-bestand (~30 MB) met 16 berichten, kortgolvige straling
omlaag (categorie 4, nummer 7), opgeteld in J/m2 vanaf de runtijd, per
kwartier tot 4 uur vooruit. Rooster 0,05 graad over Europa, 32-bit floats
(datatemplate 5.4). Dit script leest dat zelf (geen eccodes nodig) en
controleert de opbouw streng: wijkt die af, dan blijft het vorige bestand staan.

Uitvoer: per vak van 0,1 x 0,1 graad boven Nederland 16 kwartiergemiddelden in
W/m2. Bij daglicht betekent 0 in de satellietdata "geen data" (lage zon): dat
wordt null. 's Nachts (zon onder de horizon) is het 0.

Gebruik:
    KNMI_API_KEY=... python tools/knmi_nowcast.py <uitvoer.json> \\
        [--vorige <url>] [--status <status.json>]

Exitcodes: 0 = nieuw bestand, 3 = vorige behouden (zelfde run), 1 = fout
(met --vorige: dan staat het vorige bestand, als dat er is, toch in de uitvoer).
De sleutel wordt nooit gelogd of geprint.
"""

import argparse
import datetime as dt
import json
import math
import os
import re
import struct
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
import warnings

import numpy as np

DATASET = "surface_solar_irradiance"
BASIS = ("https://api.dataplatform.knmi.nl/open-data/v1/datasets/"
         f"{DATASET}/versions/1.0/files")
BRON_URL = "https://dataplatform.knmi.nl/dataset/surface-solar-irradiance-1-0"
USER_AGENT = "BanthumLabs-zonnowcast/1.0 (+https://banthumlabs.nl; info@banthumlabs.nl)"

# Vakken boven Europees Nederland (lat/lon van de linkeronderhoek, stap in graden).
# Zelfde gebied als zonverwachting.json, maar vakken van 0,1 graad.
RASTER = {"lat0": 50.7, "lon0": 3.3, "stap": 0.1, "nlat": 30, "nlon": 40}
KWARTIEREN = 16
# Verwachte opbouw van het KNMI-rooster.
GRIB_NI, GRIB_NJ = 841, 561
GRIB_LA1, GRIB_LO1, GRIB_LA2, GRIB_LO2 = 35.0, 349.0, 63.0, 31.0
GRIB_STAP = 0.05
MAX_WM2 = 1300
MAX_DOWNLOAD = 100 * 1024**2
MAX_GROOTTE = 200 * 1024
WACHTTIJDEN = (10, 30, 90)
RUN_RE = re.compile(r"__(\d{8}T\d{6})_FCST_")


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
    return dt.datetime.strptime(m.group(1), "%Y%m%dT%H%M%S").replace(tzinfo=dt.timezone.utc)


def nieuwste_run(sleutel):
    lijst = haal_json(BASIS + "?orderBy=filename&sorting=desc&maxKeys=1", sleutel)
    bestanden = lijst.get("files") or []
    if not bestanden:
        raise Fout("geen bestanden in de KNMI-lijst")
    return bestanden[0]["filename"], bestanden[0].get("size")


# ---------------------------------------------------------------- GRIB2

def _tijd(b, i):
    j, mnd, d, u, mi, s = struct.unpack(">HBBBBB", b[i:i + 7])
    return dt.datetime(j, mnd, d, u, mi, s, tzinfo=dt.timezone.utc)


def grib2_secties(b, p):
    """(lengte, {sectienummer: [(begin, lengte)]}) van het bericht dat op p begint."""
    if b[p:p + 4] != b"GRIB" or b[p + 7] != 2:
        raise Fout("geen GRIB2-bericht")
    lengte = struct.unpack(">Q", b[p + 8:p + 16])[0]
    if lengte < 20 or p + lengte > len(b) or b[p + lengte - 4:p + lengte] != b"7777":
        raise Fout("GRIB2-bericht afgebroken")
    secs, q = {}, p + 16
    while q < p + lengte - 4:
        sl = struct.unpack(">I", b[q:q + 4])[0]
        if sl < 5:
            raise Fout("ongeldige GRIB2-sectie")
        secs.setdefault(b[q + 4], []).append((q, sl))
        q += sl
    for n in (1, 3, 4, 5, 7):
        if n not in secs:
            raise Fout(f"GRIB2-sectie {n} ontbreekt")
    return lengte, secs


def lees_grib(b, run):
    """Geeft [16, nj, ni] opgeteld J/m2 (rij 0 = zuid), na strenge controle."""
    velden, p = [], 0
    while True:
        p = b.find(b"GRIB", p)
        if p < 0:
            break
        lengte, secs = grib2_secties(b, p)
        q1 = secs[1][0][0]
        if _tijd(b, q1 + 12) != run:
            raise Fout("runtijd in GRIB wijkt af van de bestandsnaam")
        q3 = secs[3][0][0]
        n_pt = struct.unpack(">I", b[q3 + 6:q3 + 10])[0]
        tmpl3 = struct.unpack(">H", b[q3 + 12:q3 + 14])[0]
        ni, nj = struct.unpack(">II", b[q3 + 30:q3 + 38])
        la1, lo1 = struct.unpack(">ii", b[q3 + 46:q3 + 54])
        la2, lo2 = struct.unpack(">ii", b[q3 + 55:q3 + 63])
        di, dj = struct.unpack(">II", b[q3 + 63:q3 + 71])
        scan = b[q3 + 71]
        if (tmpl3, ni, nj, n_pt, scan) != (0, GRIB_NI, GRIB_NJ, GRIB_NI * GRIB_NJ, 64):
            raise Fout(f"onverwacht rooster ({tmpl3}, {ni}x{nj}, scan {scan})")
        if (la1, lo1, la2, lo2, di, dj) != tuple(round(x * 1e6) for x in
                                                 (GRIB_LA1, GRIB_LO1, GRIB_LA2, GRIB_LO2, GRIB_STAP, GRIB_STAP)):
            raise Fout("onverwachte roostergrenzen")
        q4 = secs[4][0][0]
        tmpl4 = struct.unpack(">H", b[q4 + 7:q4 + 9])[0]
        cat, num = b[q4 + 9], b[q4 + 10]
        if (tmpl4, cat, num) != (8, 4, 7):
            raise Fout(f"onverwacht veld (template {tmpl4}, {cat}/{num})")
        eind = _tijd(b, q4 + 34)
        stat = b[q4 + 46]
        if stat != 1:
            raise Fout("veld is geen optelling")
        q5 = secs[5][0][0]
        n5 = struct.unpack(">I", b[q5 + 5:q5 + 9])[0]
        tmpl5 = struct.unpack(">H", b[q5 + 9:q5 + 11])[0]
        if (n5, tmpl5, b[q5 + 11]) != (n_pt, 4, 1):
            raise Fout(f"onverwachte datavorm (template {tmpl5})")
        if 6 in secs and b[secs[6][0][0] + 5] != 255:
            raise Fout("bitmap niet ondersteund")
        q7, l7 = secs[7][0]
        if l7 - 5 != 4 * n_pt:
            raise Fout("datasectie heeft de verkeerde lengte")
        velden.append((eind, np.frombuffer(b, dtype=">f4", count=n_pt, offset=q7 + 5)))
        p += lengte
    if len(velden) != KWARTIEREN:
        raise Fout(f"{len(velden)} velden (verwacht {KWARTIEREN})")
    velden.sort(key=lambda v: v[0])
    for i, (eind, _) in enumerate(velden):
        if eind != run + dt.timedelta(minutes=15 * (i + 1)):
            raise Fout(f"veld {i} eindigt op {eind:%H:%M}, verwacht run + {15 * (i + 1)} min")
    return np.stack([v.astype(np.float64).reshape(GRIB_NJ, GRIB_NI) for _, v in velden])


# ---------------------------------------------------------------- zon

def zonhoogte_deg(t, lat, lon):
    """Zonhoogte (graden), NOAA, zelfde formules als zon_stand() in het kastje."""
    uur = t.hour + t.minute / 60 + t.second / 3600
    g = 2 * math.pi / 365 * (t.timetuple().tm_yday - 1 + (uur - 12) / 24)
    eq = 229.18 * (0.000075 + 0.001868 * np.cos(g) - 0.032077 * np.sin(g) - 0.014615 * np.cos(2 * g)
                   - 0.040849 * np.sin(2 * g))
    de = (0.006918 - 0.399912 * np.cos(g) + 0.070257 * np.sin(g) - 0.006758 * np.cos(2 * g)
          + 0.000907 * np.sin(2 * g) - 0.002697 * np.cos(3 * g) + 0.00148 * np.sin(3 * g))
    ha = np.radians((uur * 60 + eq + 4 * lon) / 4 - 180)
    phi = np.radians(lat)
    cz = np.sin(phi) * np.sin(de) + np.cos(phi) * np.cos(de) * np.cos(ha)
    return 90 - np.degrees(np.arccos(np.clip(cz, -1, 1)))


def vak_waarden(opgeteld, run):
    """[vak, kwartier] W/m2 (float, NaN = geen data) uit [16, nj, ni] J/m2."""
    stap, n_lat, n_lon = RASTER["stap"], RASTER["nlat"], RASTER["nlon"]
    lats = GRIB_LA1 + GRIB_STAP * np.arange(GRIB_NJ)
    lons = (GRIB_LO1 - 360) + GRIB_STAP * np.arange(GRIB_NI)
    r = np.floor((lats - RASTER["lat0"]) / stap + 1e-6).astype(int)
    k = np.floor((lons - RASTER["lon0"]) / stap + 1e-6).astype(int)
    rij_ok = np.where((r >= 0) & (r < n_lat))[0]
    kol_ok = np.where((k >= 0) & (k < n_lon))[0]
    j0, j1, i0, i1 = rij_ok[0], rij_ok[-1] + 1, kol_ok[0], kol_ok[-1] + 1
    per_lat, per_lon = (j1 - j0) // n_lat, (i1 - i0) // n_lon
    if per_lat * n_lat != j1 - j0 or per_lon * n_lon != i1 - i0 or per_lat < 1 or per_lon < 1:
        raise Fout("vakken sluiten niet aan op de satellietpixels")

    stuk = opgeteld[:, j0:j1, i0:i1]
    wm2 = np.diff(np.concatenate([np.zeros((1,) + stuk.shape[1:]), stuk]), axis=0) / 900.0
    if not np.isfinite(wm2).all():
        wm2 = np.where(np.isfinite(wm2), wm2, np.nan)
    if (wm2 < -5).any():
        raise Fout("opgetelde straling neemt af")
    wm2 = np.clip(wm2, 0, None)
    if np.nanmax(wm2) > MAX_WM2:
        raise Fout(f"kwartiergemiddelde {np.nanmax(wm2):.0f} W/m2 boven {MAX_WM2}")

    # 0 bij daglicht = geen data (lage zon); per pixel, met de zon halverwege het kwartier.
    plat = lats[j0:j1][:, None]
    plon = lons[i0:i1][None, :]
    for q in range(KWARTIEREN):
        hoogte = zonhoogte_deg(run + dt.timedelta(minutes=15 * q + 7.5), plat, plon)
        wm2[q] = np.where((wm2[q] <= 0) & (hoogte > 0), np.nan, wm2[q])

    blok = wm2.reshape(KWARTIEREN, n_lat, per_lat, n_lon, per_lon)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # vak zonder data: NaN is de bedoeling
        gem = np.nanmean(blok, axis=(2, 4))
    return gem.reshape(KWARTIEREN, n_lat * n_lon).T


def maak_json(naam, run, waarden, mb, duur):
    lijsten = [[None if not np.isfinite(v) else int(round(v)) for v in rij] for rij in waarden]
    return {
        "bron": "KNMI satelliet-nowcast zonnestraling",
        "licentie": "CC BY 4.0",
        "bron_url": BRON_URL,
        "bestand": naam,
        "run": run.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "gemaakt": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "uitleg": ("Globale straling op een horizontaal vlak, W/m2, gemiddeld over het kwartier dat begint op "
                   "run + 15*i minuten (i = 0..15). null = geen satellietdata (lage zon). Per vak van stap x stap "
                   "graad; vak = rij * nlon + kolom, rij 0 = lat0, kolom 0 = lon0."),
        "raster": RASTER,
        "stap_min": 15,
        "kwartieren": KWARTIEREN,
        "download_mb": mb,
        "duur_s": duur,
        "waarden": lijsten,
    }


def schrijf(pad, tekst):
    os.makedirs(os.path.dirname(os.path.abspath(pad)), exist_ok=True)
    with open(pad, "w", encoding="utf-8") as f:
        f.write(tekst)


def verwerk(buf, naam):
    run = run_tijd(naam)
    waarden = vak_waarden(lees_grib(buf, run), run)
    return run, waarden


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("uitvoer")
    p.add_argument("--vorige", help="URL van het huidige zonnowcast.json")
    p.add_argument("--status", help="pad voor een klein statusbestand (ook bij een fout)")
    p.add_argument("--bestand", help="lokaal GRIB2-bestand verwerken (test, zonder KNMI)")
    args = p.parse_args()

    status = {"tijd": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")}
    vorige_tekst = None
    if args.vorige:
        try:
            with open_url(args.vorige) as r:
                vorige_tekst = r.read(MAX_GROOTTE * 2).decode("utf-8")
            json.loads(vorige_tekst)
        except (Fout, ValueError, UnicodeDecodeError) as e:
            print(f"vorige zonnowcast.json niet bruikbaar ({e})", file=sys.stderr)
            vorige_tekst = None

    try:
        begin = time.monotonic()
        if args.bestand:
            naam = os.path.basename(args.bestand)
            with open(args.bestand, "rb") as f:
                buf = f.read()
        else:
            sleutel = os.environ.get("KNMI_API_KEY", "").strip()
            if not sleutel:
                raise Fout("KNMI_API_KEY ontbreekt")
            naam, grootte = nieuwste_run(sleutel)
            status.update(nieuwste=naam)
            if vorige_tekst and json.loads(vorige_tekst).get("bestand") == naam:
                schrijf(args.uitvoer, vorige_tekst)
                status.update(resultaat="vorige behouden", run=json.loads(vorige_tekst)["run"])
                print(f"geen nieuwe run ({naam}), vorige zonnowcast.json behouden")
                return 3
            if grootte and grootte > MAX_DOWNLOAD:
                raise Fout(f"bestand groter dan {MAX_DOWNLOAD // 1024**2} MB")
            info = haal_json(f"{BASIS}/{urllib.parse.quote(naam)}/url", sleutel)
            download_url = info.get("temporaryDownloadUrl")
            if not download_url:
                raise Fout("geen download-URL van het KNMI")
            # Ondertekende tijdelijke URL: geen sleutel meesturen.
            with open_url(download_url) as r:
                buf = r.read(MAX_DOWNLOAD + 1)
            if len(buf) > MAX_DOWNLOAD:
                raise Fout(f"download groter dan {MAX_DOWNLOAD // 1024**2} MB")
        run, waarden = verwerk(buf, naam)
        duur = round(time.monotonic() - begin)
        mb = round(len(buf) / 1024**2)
        tekst = json.dumps(maak_json(naam, run, waarden, mb, duur), separators=(",", ":"))
        if len(tekst) > MAX_GROOTTE:
            raise Fout(f"zonnowcast.json is {len(tekst)} bytes (max {MAX_GROOTTE})")
        schrijf(args.uitvoer, tekst)
        leeg = int(np.isnan(waarden).sum())
        status.update(resultaat="nieuw", run=run.strftime("%Y-%m-%dT%H:%M:%SZ"), download_mb=mb, duur_s=duur,
                      bytes=len(tekst), zonder_data=leeg)
        print(f"zonnowcast.json: run {run:%Y-%m-%d %H:%M}Z, {mb} MB in {duur} s, {len(tekst)} bytes, "
              f"{leeg} waarden zonder data")
        return 0
    except Fout as e:
        status.update(resultaat="fout", fout=str(e))
        print(f"FOUT: {e}", file=sys.stderr)
    except Exception as e:  # onverwacht: vorige behouden, kort melden
        status.update(resultaat="fout", fout=f"{type(e).__name__}: {str(e)[:200]}")
        print(f"FOUT: {type(e).__name__}: {str(e)[:200]}", file=sys.stderr)
    finally:
        if args.status:
            schrijf(args.status, json.dumps(status, ensure_ascii=False, separators=(",", ":")))
    if vorige_tekst:
        schrijf(args.uitvoer, vorige_tekst)
    return 1


if __name__ == "__main__":
    sys.exit(main())
