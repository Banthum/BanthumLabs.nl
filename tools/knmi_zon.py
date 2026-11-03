"""KNMI-zondata omzetten naar data/zon.json voor de BMSmachine-kastjes.

Haalt het nieuwste bestand uit de KNMI-dataset
10-minute-in-situ-meteorological-observations (v1.0) en schrijft per station de
gemeten zonnestraling (qg) en bewolking (n, anders nc) naar een klein JSON-bestand.

Gebruik:
    KNMI_API_KEY=... python tools/knmi_zon.py <uitvoer.json> [--vorige <url>]

Met --vorige wordt eerst het huidige zon.json opgehaald. Is er bij het KNMI geen
nieuwer bestand, dan wordt niets gedownload: het vorige bestand komt in de uitvoer
en het script stopt met code 3.

Exitcodes: 0 = nieuwe data geschreven, 3 = geen nieuwe data, 1 = fout.
De sleutel wordt nooit gelogd of geprint, ook niet bij een fout.
"""

import argparse
import datetime as dt
import json
import math
import os
import sys
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request

import netCDF4
import numpy as np

BASIS = ("https://api.dataplatform.knmi.nl/open-data/v1/datasets/"
         "10-minute-in-situ-meteorological-observations/versions/1.0/files")
BRON_URL = ("https://dataplatform.knmi.nl/dataset/"
            "10-minute-in-situ-meteorological-observations-1-0")
USER_AGENT = "BanthumLabs-zon/1.0 (+https://banthumlabs.nl; info@banthumlabs.nl)"

MIN_STATIONS = 10          # minder stations met straling = fout
MAX_LEEFTIJD_MIN = 60      # meettijd ouder dan dit = fout
STRALING_MIN, STRALING_MAX = 0, 1500   # W/m²
MAX_GROOTTE = 8192         # bytes
# Alleen Europees Nederland; de Caribische stations doen er voor de kastjes niet toe.
GEBIED = {"lat": (50.0, 54.5), "lon": (2.5, 7.5)}
WACHTTIJDEN = (10, 30, 90)  # seconden tussen pogingen bij 429/5xx/netwerkfout

VERPLICHT = ("station", "stationname", "lat", "lon", "time", "qg")
BEWOLKING = ("n", "nc")     # eerste die er is en een waarde heeft


class Fout(Exception):
    pass


def haal(url, sleutel=None, max_bytes=None, herhalen=True):
    """GET met nette User-Agent en terugvallen bij 429/5xx. Foutmeldingen zonder URL of headers."""
    headers = {"User-Agent": USER_AGENT}
    if sleutel:
        headers["Authorization"] = sleutel
    wachttijden = WACHTTIJDEN if herhalen else ()
    for poging in range(len(wachttijden) + 1):
        try:
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=60) as r:
                data = r.read(max_bytes + 1 if max_bytes else -1)
                if max_bytes and len(data) > max_bytes:
                    raise Fout("antwoord groter dan verwacht")
                return data
        except urllib.error.HTTPError as e:
            status = e.code
            opnieuw = status == 429 or status >= 500
            melding = f"HTTP {status}"
        except (urllib.error.URLError, TimeoutError, ConnectionError) as e:
            opnieuw = True
            melding = f"netwerkfout ({type(e).__name__})"
        if not opnieuw or poging == len(wachttijden):
            raise Fout(melding)
        print(f"{melding}, opnieuw over {wachttijden[poging]} s", file=sys.stderr)
        time.sleep(wachttijden[poging])


def nieuwste_bestand(sleutel):
    # Sorteren op naam: de naam bevat de meettijd. Op lastModified kan een
    # later aangevuld ouder bestand bovenaan komen.
    lijst = json.loads(haal(BASIS + "?orderBy=filename&sorting=desc&maxKeys=1", sleutel))
    bestanden = lijst.get("files") or []
    if not bestanden:
        raise Fout("geen bestanden in de KNMI-lijst")
    return bestanden[0]["filename"]


def vorige_ophalen(url):
    """Huidig zon.json, of None als dat er (nog) niet is of niet te lezen is."""
    try:
        ruw = haal(url, max_bytes=64 * 1024, herhalen=False)
        data = json.loads(ruw)
        return ruw, data
    except (Fout, ValueError) as e:
        print(f"vorige zon.json niet bruikbaar ({e}), ga verder zonder", file=sys.stderr)
        return None, None


def waarde(v):
    """Getal of None bij ontbrekende waarde (masked, NaN of fill-waarde)."""
    if np.ma.is_masked(v):
        return None
    v = float(v)
    return None if math.isnan(v) else v


def omzetten(pad, bestandsnaam, nu):
    with netCDF4.Dataset(pad) as ds:
        namen = set(ds.variables)
        ontbreekt = [k for k in VERPLICHT if k not in namen]
        if ontbreekt:
            raise Fout(f"variabelen ontbreken in het KNMI-bestand: {', '.join(ontbreekt)}")
        bewolking = [k for k in BEWOLKING if k in namen]

        t = ds.variables["time"]
        if t.shape[0] < 1:
            raise Fout("geen tijd in het KNMI-bestand")
        tijd = netCDF4.num2date(t[-1], t.units, getattr(t, "calendar", "standard"),
                                only_use_cftime_datetimes=False)
        tijd = dt.datetime(tijd.year, tijd.month, tijd.day, tijd.hour, tijd.minute,
                           tzinfo=dt.timezone.utc)
        leeftijd = (nu - tijd).total_seconds() / 60
        if leeftijd > MAX_LEEFTIJD_MIN:
            raise Fout(f"KNMI-data is {leeftijd:.0f} minuten oud (max {MAX_LEEFTIJD_MIN})")

        ids = ds.variables["station"][:]
        stnamen = ds.variables["stationname"][:]
        lats = ds.variables["lat"][:]
        lons = ds.variables["lon"][:]
        qg = ds.variables["qg"][:, -1]
        wolk = [ds.variables[k][:, -1] for k in bewolking]

        stations = []
        for i in range(len(ids)):
            straling = waarde(qg[i])
            lat, lon = waarde(lats[i]), waarde(lons[i])
            if straling is None or lat is None or lon is None:
                continue
            if not (GEBIED["lat"][0] <= lat <= GEBIED["lat"][1]
                    and GEBIED["lon"][0] <= lon <= GEBIED["lon"][1]):
                continue
            if not STRALING_MIN <= straling <= STRALING_MAX:
                raise Fout(f"straling {straling} W/m² bij station {ids[i]} valt buiten "
                           f"{STRALING_MIN}-{STRALING_MAX}")
            okta = next((w for w in (waarde(a[i]) for a in wolk) if w is not None), None)
            stations.append({
                "id": str(ids[i]),
                "naam": str(stnamen[i]),
                "lat": round(lat, 2),
                "lon": round(lon, 2),
                "straling_wm2": round(straling),
                "bewolking_okta": None if okta is None else round(okta),
            })

    if len(stations) < MIN_STATIONS:
        raise Fout(f"maar {len(stations)} stations met straling (min {MIN_STATIONS})")

    return {
        "bron": "KNMI",
        "licentie": "CC BY 4.0",
        "bron_url": BRON_URL,
        "bestand": bestandsnaam,
        "tijd": tijd.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "stations": sorted(stations, key=lambda s: s["id"]),
    }


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("uitvoer")
    p.add_argument("--vorige", help="URL van het huidige zon.json")
    args = p.parse_args()

    sleutel = os.environ.get("KNMI_API_KEY", "").strip()
    if not sleutel:
        raise Fout("KNMI_API_KEY ontbreekt")

    vorige_ruw, vorige = vorige_ophalen(args.vorige) if args.vorige else (None, None)
    os.makedirs(os.path.dirname(os.path.abspath(args.uitvoer)), exist_ok=True)

    naam = nieuwste_bestand(sleutel)
    if vorige and vorige.get("bestand") == naam:
        with open(args.uitvoer, "wb") as f:
            f.write(vorige_ruw)
        print(f"geen nieuwe KNMI-data ({naam}), vorige zon.json behouden")
        return 3

    info = json.loads(haal(f"{BASIS}/{urllib.parse.quote(naam)}/url", sleutel))
    download_url = info.get("temporaryDownloadUrl")
    if not download_url:
        raise Fout("geen download-URL van het KNMI")

    with tempfile.TemporaryDirectory() as tmp:
        pad = os.path.join(tmp, "knmi.nc")
        # De tijdelijke download-URL is al ondertekend: hier geen sleutel meesturen.
        with open(pad, "wb") as f:
            f.write(haal(download_url, max_bytes=50 * 1024 * 1024))
        zon = omzetten(pad, naam, dt.datetime.now(dt.timezone.utc))

    tekst = json.dumps(zon, ensure_ascii=False, separators=(",", ":"))
    grootte = len(tekst.encode("utf-8"))
    if grootte > MAX_GROOTTE:
        raise Fout(f"zon.json is {grootte} bytes (max {MAX_GROOTTE})")

    with open(args.uitvoer, "w", encoding="utf-8") as f:
        f.write(tekst)
    print(f"zon.json geschreven: {zon['tijd']}, {len(zon['stations'])} stations, {grootte} bytes")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Fout as e:
        print(f"FOUT: {e}", file=sys.stderr)
        sys.exit(1)
