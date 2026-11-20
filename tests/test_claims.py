"""Claims: jede Zahl der README (Abschnitte „Befunde und Korrekturen“ und „Befunde“) steht hier, mit Bedingung und Band (Tabelle „Zahl → Test“ der README).

Band: Zahlen aus eingefrorenen Messungen oder aus Ziehungen mit SplitMix64 werden auf drei Nachkommastellen verglichen mit Toleranz `TOL` (gerundete Angabe + Rechenrauschen der Plattform);
exakte Zählungen und Brüche exakt. Die Zeilenmengen und Zähleinheiten stehen in den Kommentaren."""

import collections

import numpy as np
import pytest

import shn_data as DATA
import shn_experiments as E
import shn_toy as T

TOL = 6e-4


def _l(label_beginn, n=None, rauschen=None):
    zs = [z for z in DATA.learn()["toy"]["zeilen"] if z["lernverfahren"].startswith(label_beginn) and (n is None or z["n"] == n) and (rauschen is None or z["rauschen"] == rauschen)]
    assert len(zs) == 1, (label_beginn, n, rauschen)
    return zs[0]


def _l_zweit(label_beginn, n, rauschen):
    return next(z for z in DATA.learn(True)["toy"]["zeilen"] if z["lernverfahren"].startswith(label_beginn) and z["n"] == n and z["rauschen"] == rauschen)


def _gleich(werte, soll):
    assert werte == pytest.approx(soll, abs=TOL), (werte, soll)


def _spanne(werte, lo, hi, tol=TOL):
    assert min(werte) == pytest.approx(lo, abs=tol) and max(werte) == pytest.approx(hi, abs=tol), (min(werte), max(werte), lo, hi)


# ------------------------------------------------------------------------------------------------ B1: Baumformen
def test_b1_klassen_wurzeln_und_wahrheit():
    r = E.exp1(0.9, 0, "pfadabhängig")
    assert r["n_formen"] == 131 and [(round(w[1], 6), c) for w, c in r["klassen"]] == [(0.0, 97), (0.1, 8), (0.2, 26)]
    nach_tiefe = collections.Counter(b.tiefe() for b in E.formen(0.9))
    assert nach_tiefe == {1: 1, 2: 10, 3: 120}
    c = collections.Counter()
    for b, w in zip(E.formen(0.9), E.wichtigkeiten_pfad(0.9)):
        c[(T.wurzel_merkmal(b), round(w[1], 3))] += 1
    assert c[(0, 0.0)] == 81 and sum(v for (wz, _), v in c.items() if wz == 0) == 81      # Wurzel x1: 81 Bäume, alle mit Beitrag 0
    assert c[(1, 0.2)] == 25 and sum(v for (wz, _), v in c.items() if wz == 1) == 25      # Wurzel x2: 25 Bäume, alle 0,200
    _gleich(r["wahrheit"]["interventional"], [0.5, 0, 0])
    _gleich(r["wahrheit"]["konditional"], [0.34, 0.2, 0])
    assert sum(1 for b, w in zip(E.formen(0.9), E.wichtigkeiten_pfad(0.9)) if abs(float(T.masse_x2_vor_x1(b)) / 5 - w[1]) < 1e-12) == 131     # Regel Masse / 5 (Gleitkomma; Brüche in test_handbeispiele)
    mini = E.mini_beispiel()
    assert [str(x) for x in mini["A"]["phi"]] == ["1/2", "0"] and [str(x) for x in mini["B"]["phi"]] == ["3/10", "1/5"]


# ------------------------------------------------------------------------------------------------ B2: Lernverfahren im Toy ohne Rauschen
def test_b2_toy_ohne_rauschen():
    for name in ("gieriger Baum", "Boosting"):
        for n in (400, 4000):
            z = _l(name, n, 0.0)
            _gleich(z["pfad"], [0.5, 0, 0])
            _gleich(z["interv"], [0.5, 0, 0])
            assert round(z["anteil_phi2"] * 30) == 0
    z = _l("Zufallsbaum", 400, 0.0)
    _gleich(z["pfad"], [0.420, 0.100, 0.009])
    assert round(z["anteil_phi2"] * 30) == 18 and _l("Zufallsbaum", 4000, 0.0)["pfad"][1] == pytest.approx(0.1, abs=TOL)
    z = _l("Wald 60 Bäume, 1 Merkmal", 400, 0.0)
    _gleich(z["pfad"], [0.421, 0.099, 0.003])
    _gleich(z["interv"], [0.5, 0, 0])
    assert round(z["anteil_phi2"] * 30) == 30
    z = _l("Wald 60 Bäume, 2 Merkmale", 400, 0.0)
    _gleich(z["pfad"], [0.447, 0.066, 0.001])
    assert round(z["anteil_phi2"] * 30) == 30
    for name in ("Zufallsbaum", "Wald 60 Bäume, 1 Merkmal", "Wald 60 Bäume, 2 Merkmale"):         # n = 4000 gibt dieselben Zahlen auf zwei Nachkommastellen
        a, b = _l(name, 400, 0.0), _l(name, 4000, 0.0)
        assert max(abs(x - y) for x, y in zip(a["pfad"], b["pfad"])) < 0.01 + 1e-9


def test_befunde_und_korrekturen_modellwerte_nicht_ueberall_exakt_f():
    for name in ("Wald 60 Bäume, 1 Merkmal", "Wald 60 Bäume, 2 Merkmale"):
        assert _l(name, 400, 0.0)["abweichung_von_f"] == pytest.approx(1 / 60, abs=1e-4) and _l(name, 4000, 0.0)["abweichung_von_f"] == 0
    for name in ("gieriger Baum", "Zufallsbaum"):
        assert _l(name, 400, 0.0)["abweichung_von_f"] == 0
    assert 0 < _l("Boosting", 400, 0.0)["abweichung_von_f"] <= 2e-5 and 0 < _l("Boosting", 4000, 0.0)["abweichung_von_f"] <= 2e-5


# ------------------------------------------------------------------------------------------------ B3: mit Rauschen
def test_b3_toy_mit_rauschen():
    z = _l("gieriger Baum", 400, 0.5)
    _gleich(z["pfad"], [0.496, 0.031, 0.026])
    _gleich(z["interv"], [0.496, 0.043, 0.027])
    assert round(z["anteil_phi2"] * 30) == 22
    z = _l("gieriger Baum", 4000, 0.5)
    _gleich(z["pfad"], [0.500, 0.008, 0.008])
    _gleich(z["interv"], [0.501, 0.011, 0.008])
    assert round(z["anteil_phi2"] * 30) == 0
    z = _l("Wald 60 Bäume, 1 Merkmal", 4000, 0.5)
    _gleich(z["pfad"], [0.422, 0.097, 0.009])
    _gleich(z["interv"], [0.502, 0.012, 0.008])
    assert round(z["anteil_phi2"] * 30) == 30


# ------------------------------------------------------------------------------------------------ B4: Vehikel
def test_b4_vehikel_ziel_vorlauf():
    v = DATA.learn()["vehikel"]
    zeilen = {z["lernverfahren"]: z for z in v["zeilen"]}
    assert v["n_je_menge"] == 25 and len(v["mengen"]) == 5 and len(v["modell_seeds"]) == 5

    def art(z, key):
        return [m[0] for m in z[key]]
    for name, r2 in (("gieriger Baum, Tiefe 6", 0.9998), ("Wald 60 Bäume, alle Merkmale je Schnitt", 1.0000), ("Boosting 200 Bäume, Tiefe 3", 1.0000)):
        z = zeilen[name]
        assert z["r2"] == pytest.approx(r2, abs=1e-4) and max(art(z, "interv_je_menge") + art(z, "pfad_je_menge")) <= 0.0012
    z = zeilen["Wald 60 Bäume, 0,5 der Merkmale je Schnitt"]
    assert z["r2"] == pytest.approx(0.9980, abs=1e-4)
    _spanne(art(z, "interv_je_menge"), 0.540, 0.639)
    _spanne(art(z, "pfad_je_menge"), 0.711, 0.813)
    luecke05 = [p - i for p, i in zip(art(z, "pfad_je_menge"), art(z, "interv_je_menge"))]
    z1 = zeilen["Wald 60 Bäume, 1 Merkmal je Schnitt"]
    assert z1["r2"] == pytest.approx(0.9940, abs=1e-4)
    _spanne(art(z1, "interv_je_menge"), 0.672, 0.797)
    _spanne(art(z1, "pfad_je_menge"), 0.859, 0.983)
    luecke1 = [p - i for p, i in zip(art(z1, "pfad_je_menge"), art(z1, "interv_je_menge"))]
    assert min(luecke05 + luecke1) >= 0.125 and max(luecke05 + luecke1) <= 0.21 + 1e-3        # „0,13 bis 0,21“ (gerundet)
    w = v["wahrheit_je_menge"]
    assert all(m[j] == 0 for m in w["interventional"] for j in range(4)) and all(m[0] == 0 for m in w["interventional"])
    _spanne([m[0] for m in w["konditional"]], 0.94, 1.18, tol=0.006)                           # „konditional Art 0,94 bis 1,18“


# ------------------------------------------------------------------------------------------------ B5, B6: Kosten
def test_b5_abweichung_algorithmus_2_in_den_zellen():
    assert max(z["abweichung_max"] for z in DATA.costs()["tabelle"]) < 2e-15 and len(DATA.costs()["tabelle"]) == 15


def test_b6_kostentabelle_anker():
    zeilen = {(z["d"], z["tiefe"]): z for z in E.kosten_tabelle()}
    a, b = zeilen[(14, 3)], zeilen[(6, 8)]
    assert a["aufzaehlung"][1:] == (133120, 133120) and (a["alg2"][1], a["alg2"][2]) == (96, 121) and a["n_baeume"] == 4
    assert (b["alg2"][1], b["alg2"][2]) == (6084, 9509) and (b["aufzaehlung"][1], b["aufzaehlung"][2]) == (5074, 5828) and b["alg2_mehr"] == 8 and b["n_baeume"] == 8
    assert zeilen[(6, 6)]["alg2_mehr"] == 1 and zeilen[(6, 6)]["n_baeume"] == 8
    assert (zeilen[(8, 8)]["verhaeltnis"][0], zeilen[(8, 8)]["verhaeltnis"][1]) == pytest.approx((1.76, 3.61), abs=0.006)
    assert all(z["alg2_mehr"] == 0 for k, z in zeilen.items() if k not in ((6, 6), (6, 8)))      # sonst nie mehr (Erstmessung)
    vk = [(v["d"], v["aufzaehlung"], v["alg2"]) for v in DATA.costs()["vollstaendig"]]
    assert all(a_ == 3 ** (d + 1) - 2 ** (d + 1) and s_ == sum(2 ** k * (1 + k) for k in range(d + 1)) + 2 ** d * d * d for d, a_, s_ in vk) and len(vk) == 8
    assert E.exp3("vollständig (Tiefe = d)", 10)["besuche"] == 175099 and E.exp3("vollständig (Tiefe = d)", 10)["schritte"] == 122881


# ------------------------------------------------------------------------------------------------ B7: Ensembles
def test_b7_abstand_am_ensemble():
    z = {e["modell"]: e for e in E.ensembles()}
    b, w5, wa = z["Boosting 200 Bäume, Tiefe 3"], z["Wald 60 Bäume, 0,5 der Merkmale je Schnitt"], z["Wald 60 Bäume, alle Merkmale je Schnitt"]
    assert (b["r2"], w5["r2"], wa["r2"]) == pytest.approx((0.927, 0.894, 0.894), abs=0.001)
    assert b["median"] == pytest.approx((0.065, 0.076), abs=TOL) and w5["median"] == pytest.approx((0.150, 0.199), abs=TOL) and wa["median"] == pytest.approx((0.130, 0.156), abs=TOL)
    assert b["rang1"] == pytest.approx((0.013, 0.087), abs=TOL) and w5["rang1"] == pytest.approx((0.033, 0.140), abs=TOL) and wa["rang1"] == pytest.approx((0.053, 0.100), abs=TOL)
    assert b["vorzeichen"] == w5["vorzeichen"] == wa["vorzeichen"] == (0.0, 0.0)


# ------------------------------------------------------------------------------------------------ B8, B9, B10: KernelSHAP
def _k(d, modell, m, gepaart, z):
    r = E.kernel_experiment(d, modell, m, gepaart, z)
    return min(r["median_je_menge"]), max(r["median_je_menge"]), r


def test_b8_kernelshap_d6_24_koalitionen():
    erwartet = {  # Modell: (einfach lo, hi, gepaart lo, hi)
        "Boosting Tiefe 1": (0.0, 0.0, 0.0, 0.0), "Boosting Tiefe 2": (0.038, 0.085, 0.0, 0.0), "Boosting Tiefe 3": (0.083, 0.168, 0.005, 0.019),
        "Boosting Tiefe 4": (0.089, 0.146, 0.013, 0.043), "Wald Tiefe 2": (0.011, 0.033, 0.0, 0.0), "Wald unbegrenzt": (0.127, 0.223, 0.037, 0.076), "Netz": (0.091, 0.143, 0.018, 0.042)}
    for modell, (e_lo, e_hi, g_lo, g_hi) in erwartet.items():
        lo, hi, r = _k(6, modell, 24, False, 10)
        assert (lo, hi) == pytest.approx((e_lo, e_hi), abs=TOL), modell
        lo, hi, rg = _k(6, modell, 24, True, 10)
        assert (lo, hi) == pytest.approx((g_lo, g_hi), abs=TOL), modell
        assert rg["auswertungen"] == 720 and rg["auswertungen_aufzaehlung"] == 1920 and rg["anteil"] == 0.375 and rg["singulaer"] == 0 and rg["n_mengen"] == 5 and rg["zeilen_je_menge"] == 10
    assert _k(6, "Wald unbegrenzt", 24, True, 10)[2]["q90"] == pytest.approx(0.182, abs=TOL)           # der Median verdeckt den Tail
    assert max(_k(6, "Boosting Tiefe 2", 24, True, 10)[:2]) < 1e-9 and max(_k(6, "Boosting Tiefe 1", 24, True, 10)[:2]) < 1e-9   # exakt bis auf Rundung


def test_b9_kernelshap_d10():
    erwartet = {"Boosting Tiefe 3": ((0.011, 0.021), (0.068, 0.155), (0.013, 0.023)), "Wald unbegrenzt": ((0.039, 0.076), (0.104, 0.279), (0.022, 0.040)),
                "Netz": ((0.036, 0.053), (0.119, 0.223), (0.019, 0.037))}
    for modell, (gep, einf, einf1280) in erwartet.items():
        lo, hi, r = _k(10, modell, 40, True, 6)
        assert (lo, hi) == pytest.approx(gep, abs=TOL), modell
        assert r["auswertungen"] == 1200 and r["auswertungen_aufzaehlung"] == 30720 and round(100 * r["anteil"], 1) == 3.9
        assert r["singulaer"] == 1 and r["n_ziehungen"] == 30                                      # „1 von 30 Ziehungen singulär“ (Befunde und Korrekturen)
        assert _k(10, modell, 40, False, 6)[:2] == pytest.approx(einf, abs=TOL)
        lo, hi, r1280 = _k(10, modell, 1280, False, 6)
        assert (lo, hi) == pytest.approx(einf1280, abs=TOL) and r1280["anteil"] == 1.25 and r1280["auswertungen"] > r1280["auswertungen_aufzaehlung"]


def test_b10_orakel_und_ein_paar_singulaer():
    for d in (6, 10):
        for z in DATA.games(d)["modelle"]:
            assert E.orakel_alle_koalitionen(d, z["modell"]) < 1e-9
    assert E.kernel_experiment(6, "Boosting Tiefe 3", 2, True, 2)["singulaer"] == 10


# ------------------------------------------------------------------------------------------------ B11, B12: Ordnung
def test_b11_moebius_ordnung():
    z = {x["modell"]: x for x in E.ordnung_tabelle()}
    assert [z[f"Boosting Tiefe {k}"]["ordnung"] for k in (1, 2, 3, 4)] == [1, 2, 3, 4] and z["Wald Tiefe 2"]["ordnung"] == 2 and z["Wald unbegrenzt"]["ordnung"] == 6 and z["Netz"]["ordnung"] == 6
    assert z["Boosting Tiefe 1"]["anteile"][0] == pytest.approx(1.0)
    assert z["Boosting Tiefe 2"]["anteile"][:2] == pytest.approx([0.945, 0.055], abs=1e-3) and z["Boosting Tiefe 3"]["anteile"][:2] == pytest.approx([0.851, 0.144], abs=1e-3)
    assert z["Boosting Tiefe 4"]["anteile"][:2] == pytest.approx([0.811, 0.166], abs=1e-3)
    assert sum(z["Wald unbegrenzt"]["anteile"][2:]) == pytest.approx(0.061, abs=1e-3) and sum(z["Netz"]["anteile"][2:]) == pytest.approx(0.019, abs=1e-3)
    assert DATA.games(6)["instanzen_je_menge"] == 10 and len(DATA.games(6)["modell_seeds"]) == 5


def _anzahl_exakt(r):
    return round(sum(r["anteile"]) * r["n_spiele"])


def test_b12_zufallsspiele_zaehlungen():
    for d, ms in ((6, (4, 5, 6, 8, 12, 16)), (10, (8, 9, 12, 16, 20))):
        for m in ms:
            r = E.zufallsspiele(d, 3, m)
            assert _anzahl_exakt(r) == 0, (d, m)                                               # Ordnung 3: in keinem der 200 Spiele exakt
            for o in (1, 2):
                r = E.zufallsspiele(d, o, m)
                assert _anzahl_exakt(r) + r["singulaer"] == 200, (d, o, m)                      # Ordnung ≤ 2: exakt genau dann, wenn nicht singulär
    soll6 = {5: 61, 8: 170, 12: 196, 16: 199}
    soll10 = {9: 36, 12: 147, 16: 192, 20: 198}
    for m, k in soll6.items():
        assert abs(_anzahl_exakt(E.zufallsspiele(6, 2, m)) - k) <= 2, m
    for m, k in soll10.items():
        assert abs(_anzahl_exakt(E.zufallsspiele(10, 2, m)) - k) <= 2, m
    assert E.zufallsspiele(6, 2, 4)["singulaer"] == 200
    r8 = E.zufallsspiele(6, 2, 8)
    assert min(r8["anteile"]) == pytest.approx(0.82, abs=0.03) and max(r8["anteile"]) == pytest.approx(0.88, abs=0.03)     # Bandbreite zwischen den fünf Gruppen („0,82 bis 0,88“)
    # Zweitmessung (andere Zufalls-Seeds der Spiele und der Ziehungen): Zahlen der README
    for m, k in {5: 67, 8: 171, 12: 198, 16: 200}.items():
        assert abs(_anzahl_exakt(E.zufallsspiele(6, 2, m, salz=1)) - k) <= 2, m
    for m, k in {9: 32, 12: 143, 16: 191, 20: 198}.items():
        assert abs(_anzahl_exakt(E.zufallsspiele(10, 2, m, salz=1)) - k) <= 2, m
    r8z = E.zufallsspiele(6, 2, 8, salz=1)
    assert min(r8z["anteile"]) == pytest.approx(0.82, abs=0.03) and max(r8z["anteile"]) == pytest.approx(0.93, abs=0.03)
    assert abs(np.sqrt(0.85 * 0.15 / 40) - 0.06) < 0.004                                        # Standardfehler je Gruppe bei p = 0,85 („etwa 0,06“)


# ------------------------------------------------------------------------------------------------ Zweitmessung (andere Zufalls-Seeds)
def test_zweitmessung_befunde_und_korrekturen():
    zk = {(z["d"], z["tiefe"]): z for z in DATA.costs(True)["tabelle"]}
    z = zk[(6, 8)]
    assert sum(1 for a, b in zip(z["aufzaehlung"], z["alg2"]) if b > a) == 6 and len(z["alg2"]) == 8
    r = [a / b for a, b in zip(z["aufzaehlung"], z["alg2"])]
    assert (min(r), max(r)) == pytest.approx((0.55, 1.05), abs=0.006)
    assert round(_l_zweit("gieriger Baum", 4000, 0.5)["anteil_phi2"] * 30) == 1
    ens = DATA.costs(True)["ensembles"]["zeilen"]
    assert all(e["vorzeichen_anders_je_menge"] == [0.0] * 5 for e in ens if e["modell"].startswith("Boosting"))
    for e in ens:
        if e["modell"].startswith("Wald"):
            assert sorted(e["vorzeichen_anders_je_menge"])[-1] == pytest.approx(1 / 150, abs=1e-5) and sorted(e["vorzeichen_anders_je_menge"])[-2] == 0.0
    assert DATA.costs()["ensembles"]["n_je_menge"] == 30 and 5 * 30 == 150                       # fünf Zufalls-Seeds mal 30 Zeilen: 150 Zeilen je Menge (Erstmessung: 0 anders)
    for e in DATA.costs()["ensembles"]["zeilen"]:
        assert e["vorzeichen_anders_je_menge"] == [0.0] * 5
    # Zweitmessung Vehikel: Lücke kleiner als in der Erstmessung, Wald mit der Hälfte der Merkmale
    v = {e["lernverfahren"]: e for e in DATA.learn(True)["vehikel"]["zeilen"]}
    w = v["Wald 60 Bäume, 0,5 der Merkmale je Schnitt"]
    lo = [p[0] - i[0] for p, i in zip(w["pfad_je_menge"], w["interv_je_menge"])]
    assert 0.08 <= min(lo) and max(lo) <= 0.19
    # Ordnung gleich Baumtiefe auch in der Zweitmessung
    assert [x["ordnung"] for x in E.ordnung_tabelle(True)] == [1, 2, 3, 4, 2, 6, 6]
    # KernelSHAP der Zweitmessung: gepaart Boosting Tiefe 1 und 2 exakt
    zw = E.kernel_experiment(6, "Boosting Tiefe 2", 24, True, 10, True)
    assert max(zw["median_je_menge"]) < 1e-9


def test_befunde_und_korrekturen_kernelshap_je_instanzmenge_gemeinsame_ziehungen():
    """Die Ziehungen gelten für alle Zeilen einer Menge gemeinsam (Abweichung vom Vorgehen der Vormessung): im Code ein Aufruf von masken_ziehen je (Menge, Ziehung)."""
    import inspect
    quelle = inspect.getsource(E.kernel_experiment)
    assert quelle.count("masken_ziehen(") == 1 and quelle.index("masken_ziehen(") < quelle.index("for i in range(V.shape[1])")
