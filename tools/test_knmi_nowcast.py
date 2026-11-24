"""Tests voor knmi_nowcast.py met een zelfgemaakt GRIB2-bestand (geen netwerk).

    python -m unittest tools/test_knmi_nowcast.py
"""

import datetime as dt
import os
import struct
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import knmi_nowcast as N  # noqa: E402

RUN = dt.datetime(2026, 11, 24, 12, 0, tzinfo=dt.timezone.utc)
NAAM = "SEVIR_OPER_R__CPP_AODC_L2__20261124T120000_FCST_20261124T160000_0002_SUB_europapa.grb2"


def sectie(n, inhoud):
    return struct.pack(">IB", 5 + len(inhoud), n) + inhoud


def tijd(t):
    return struct.pack(">HBBBBB", t.year, t.month, t.day, t.hour, t.minute, t.second)


def bericht(eind, veld, tmpl4=8, cat=4):
    s1 = sectie(1, b"\x00" * 7 + tijd(RUN) + b"\x00\x00")
    s3 = bytearray(sectie(3, b"\x00" * 67))
    struct.pack_into(">I", s3, 6, N.GRIB_NI * N.GRIB_NJ)
    struct.pack_into(">H", s3, 12, 0)
    struct.pack_into(">II", s3, 30, N.GRIB_NI, N.GRIB_NJ)
    struct.pack_into(">ii", s3, 46, round(N.GRIB_LA1 * 1e6), round(N.GRIB_LO1 * 1e6))
    struct.pack_into(">ii", s3, 55, round(N.GRIB_LA2 * 1e6), round(N.GRIB_LO2 * 1e6))
    struct.pack_into(">II", s3, 63, round(N.GRIB_STAP * 1e6), round(N.GRIB_STAP * 1e6))
    s3[71] = 64
    s4 = bytearray(sectie(4, b"\x00" * 53))
    struct.pack_into(">H", s4, 7, tmpl4)
    s4[9], s4[10] = cat, 7
    s4[34:41] = tijd(eind)
    s4[46] = 1
    s5 = bytearray(sectie(5, b"\x00" * 7))
    struct.pack_into(">I", s5, 5, N.GRIB_NI * N.GRIB_NJ)
    struct.pack_into(">H", s5, 9, 4)
    s5[11] = 1
    s6 = sectie(6, b"\xff")
    s7 = sectie(7, veld.astype(">f4").tobytes())
    romp = s1 + bytes(s3) + bytes(s4) + bytes(s5) + s6 + s7
    lengte = 16 + len(romp) + 4
    return b"GRIB\x00\x00\x00\x02" + struct.pack(">Q", lengte) + romp + b"7777"


def bestand(wm2=300.0, aanpassen=None, **kw):
    """16 berichten met constante straling wm2; aanpassen(q, veld) kan een veld wijzigen."""
    delen, opgeteld = [], np.zeros((N.GRIB_NJ, N.GRIB_NI))
    for q in range(N.KWARTIEREN):
        stap = np.full((N.GRIB_NJ, N.GRIB_NI), wm2 * 900.0)
        if aanpassen:
            aanpassen(q, stap)
        opgeteld = opgeteld + stap
        delen.append(bericht(RUN + dt.timedelta(minutes=15 * (q + 1)), opgeteld.ravel(), **kw))
    return b"".join(delen)


class TestNowcast(unittest.TestCase):
    def test_constante_straling(self):
        run, w = N.verwerk(bestand(300.0), NAAM)
        self.assertEqual(run, RUN)
        self.assertEqual(w.shape, (N.RASTER["nlat"] * N.RASTER["nlon"], N.KWARTIEREN))
        # 12-13 UTC eind november: zon boven NL, dus alles 300.
        self.assertTrue(np.allclose(w[:, :4], 300.0))

    def test_nul_bij_daglicht_is_geen_data(self):
        def nul(q, veld):
            if q == 0:
                veld[:] = 0
        _, w = N.verwerk(bestand(300.0, nul), NAAM)
        self.assertTrue(np.isnan(w[:, 0]).all())
        self.assertTrue(np.allclose(w[:, 1], 300.0))

    def test_nul_na_zonsondergang_blijft_nul(self):
        # Kwartier 15 = 15:45-16:00 UTC: in heel NL is de zon dan onder.
        def nacht(q, veld):
            if q == 15:
                veld[:] = 0
        _, w = N.verwerk(bestand(300.0, nacht), NAAM)
        self.assertTrue((w[:, 15] == 0).all())

    def test_een_pixel_zonder_data_telt_niet_mee(self):
        lat_i = round((51.95 - N.GRIB_LA1) / N.GRIB_STAP)
        lon_i = round((5.85 - (N.GRIB_LO1 - 360)) / N.GRIB_STAP)

        def gat(q, veld):
            veld[:] = 300 * 900.0
            veld[lat_i, lon_i] = 0
        _, w = N.verwerk(bestand(300.0, gat), NAAM)
        self.assertFalse(np.isnan(w).any())
        self.assertTrue(np.allclose(w[:, :4], 300.0))

    def test_json_heeft_null(self):
        def nul(q, veld):
            if q == 0:
                veld[:] = 0
        run, w = N.verwerk(bestand(300.0, nul), NAAM)
        j = N.maak_json(NAAM, run, w, 30, 1)
        self.assertIsNone(j["waarden"][0][0])
        self.assertEqual(j["waarden"][0][1], 300)
        self.assertEqual(j["run"], "2026-11-24T12:00:00Z")

    def test_ander_veld_geweigerd(self):
        with self.assertRaises(N.Fout):
            N.verwerk(bestand(300.0, cat=5), NAAM)

    def test_afwijkende_runtijd_geweigerd(self):
        with self.assertRaises(N.Fout):
            N.verwerk(bestand(300.0), NAAM.replace("T120000_FCST", "T121500_FCST"))

    def test_te_weinig_berichten_geweigerd(self):
        b = bestand(300.0)
        with self.assertRaises(N.Fout):
            N.verwerk(b[: b.rfind(b"GRIB")], NAAM)

    def test_onmogelijk_hoge_straling_geweigerd(self):
        with self.assertRaises(N.Fout):
            N.verwerk(bestand(2000.0), NAAM)


if __name__ == "__main__":
    unittest.main()
