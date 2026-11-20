"""Verdikte an von Hand gesetzten Werten (DEMO-PLAYBOOK §12): jede Schwelle der Art und jede Zahl im Text steht hier. Die Verdikte sind reine Funktionen (Art, Text)."""

import shn_evaluation as ev


def test_zahlformate_mit_komma_und_geschuetztem_tausenderzeichen():
    assert ev.de(0.0996, 3) == "0,100" and ev.de(3.0, 2) == "3,00"
    assert ev.tausend(1247744) == "1 247 744" and ev.tausend(95) == "95"
    assert ev.spanne(0.005, 0.0204) == "0,005 bis 0,020" and ev.spanne(0.1, 0.1004) == "0,100"
    assert ev.prozent(0.375) == "37,5 %"


# ----------------------------------------------------------------------------------------------- Experiment 1
def _r1(klassen, w2, funktional="pfadabhängig", q=0.9, wurzel=1, masse=1.0):
    return {"n_formen": 131, "klassen": klassen, "funktional": funktional, "q": q, "wichtigkeit": [0.34, w2, 0.0], "wahrheit": {"konditional": [0.34, 0.2, 0.0], "interventional": [0.5, 0, 0]},
            "wurzel": wurzel, "masse_x2_vor_x1": masse}


DREI = [((0.5, 0.0, 0.0), 97), ((0.42, 0.1, 0.0333), 8), ((0.34, 0.2, 0.0), 26)]


def test_formen_verdikt_nennt_die_drei_klassen_und_die_regel_bei_q_09():
    kind, text = ev.verdict_formen(_r1(DREI, 0.2))
    assert kind == "warning"
    assert "von 131 Baumformen geben 97 dem nie gelesenen x₂ den Beitrag 0, 8 den Wert 0,100, 26 den konditionalen Wert 0,200 (Verteilung q = 0,90)" in text
    assert "Gewählt: x₂ bekommt hier 0,200" in text and "ein Fünftel der Masse" in text and "hier 1,00 · 1/5 = 0,200" in text
    assert "Liegt x₁ an der Wurzel" not in text                                              # Wurzel x₂: der Zusatz gilt nur für Wurzel x₁


def test_formen_verdikt_regel_nur_bei_q_09_und_info_bei_null():
    kind, text = ev.verdict_formen(_r1(DREI, 0.0, q=0.75, wurzel=0, masse=0.0))
    assert kind == "info" and "Fünftel" not in text and "Verteilung q = 0,75" in text and "bekommt x₂ immer 0" in text


def test_formen_verdikt_gleich_bei_q_05_und_anderem_funktional():
    kind, text = ev.verdict_formen(_r1([((0.5, 0.0, 0.0), 131)], 0.0, q=0.5))
    assert kind == "info" and "geben alle 131 Baumformen dieselben Beiträge" in text and "q = 0,50" in text
    kind, text = ev.verdict_formen(_r1([((0.5, 0.0, 0.0), 131)], 0.0, funktional="interventional"))
    assert kind == "info" and "Interventional: alle 131 Baumformen geben denselben Beitrag (x₂: 0,000)" in text and "nur das pfadabhängige" in text


# ----------------------------------------------------------------------------------------------- Experiment 2
def _toy(rauschen, pfad2, interv2, anteil, abw=0.0, n=400):
    return {"welt": "toy", "zeile": {"rauschen": rauschen, "pfad": [0.4, pfad2, 0.0], "interv": [0.5, interv2, 0.0], "anteil_phi2": anteil, "n_seeds": 30, "n": n, "abweichung_von_f": abw},
            "wahrheit": {"konditional": [0.34, 0.2, 0.0], "interventional": [0.5, 0.0, 0.0]}}


def test_lernen_toy_ohne_rauschen_warnt_ab_phi2_ueber_schwelle():
    kind, text = ev.verdict_lernen(_toy(0.0, 0.099, 0.0, 1.0))
    assert kind == "warning" and "Modellwerte auf den acht Punkten gleich f" in text
    assert "interventional 0,000, pfadabhängig 0,099 (Mittel über 30 Zufalls-Seeds; |φ₂| > 0,02 in 30 von 30)" in text and "konditional 0,200" in text
    kind, text = ev.verdict_lernen(_toy(0.0, 0.0, 0.0, 0.0))
    assert kind == "success" and "in 0 von 30" in text
    assert "um höchstens 0,0167 entfernt" in ev.verdict_lernen(_toy(0.0, 0.099, 0.0, 1.0, abw=0.0167))[1]


def test_lernen_toy_mit_rauschen_ist_eine_info():
    kind, text = ev.verdict_lernen(_toy(0.5, 0.031, 0.043, 0.73))
    assert kind == "info" and "Mit Rauschen 0,5 ist das Modell nicht mehr f" in text and "interventional 0,043, pfadabhängig 0,031" in text and "in 22 von 30" in text


def _veh(art_i, art_p, name="Wald 60 Bäume, 0,5 der Merkmale je Schnitt"):
    return {"welt": "vehikel", "zeile": {"lernverfahren": name, "r2": 0.998}, "art_interv": art_i, "art_pfad": art_p, "luecke_art": (art_p[0] - art_i[0], art_p[1] - art_i[1]),
            "wahrheit": {"konditional": [[1.18, 0, 0.14, 0, 1.67], [0.94, 0, 0.1, 0, 1.6]]}}


def test_lernen_vehikel_liest_die_art_nicht_oder_wirklich():
    kind, text = ev.verdict_lernen(_veh((0.0, 0.0), (0.001, 0.001)))
    assert kind == "success" and "das Modell liest die Art nicht" in text and "konditional 0,94 bis 1,18" in text
    kind, text = ev.verdict_lernen(_veh((0.54, 0.64), (0.71, 0.81)))
    assert kind == "warning" and "interventional 0,540 bis 0,640, pfadabhängig 0,710 bis 0,810" in text and "die Lücke zwischen den Funktionalen ist 0,17;" in text
    assert "R² 0,9980 gegen g" in text and "fünf Instanzmengen zu je 25 Zeilen" in text


# ----------------------------------------------------------------------------------------------- Experiment 3
def test_kosten_live_wenn_algorithmus_2_weniger_oder_mehr_braucht():
    r = {"d": 6, "tiefe": 3, "blaetter": 8, "besuche": 576, "schritte": 96, "abweichung": 2.2e-16}
    kind, text = ev.verdict_kosten_live(r)
    assert kind == "info" and "d = 6, Tiefe 3, 8 Blätter" in text and "Aufzählung 576 Besuche, Algorithmus 2 96 Schritte, also 6,00-mal weniger" in text and "2,2e-16" in text
    r.update(besuche=5, schritte=7, d=1, tiefe=1, blaetter=2)
    kind, text = ev.verdict_kosten_live(r)
    assert kind == "warning" and "also 1,40-mal so viele wie die Aufzählung" in text


def test_kosten_tabelle_anker():
    zeilen = [{"d": 14, "tiefe": 3, "n_baeume": 4, "aufzaehlung": (133120.0, 133120, 133120), "alg2": (115.0, 96, 121), "alg2_mehr": 0},
              {"d": 6, "tiefe": 8, "n_baeume": 8, "aufzaehlung": (5360.0, 5074, 5828), "alg2": (7398.0, 6084, 9509), "alg2_mehr": 8}]
    kind, text = ev.verdict_kosten_tabelle(zeilen)
    assert kind == "info"
    assert "d = 14, Tiefe 3: Aufzählung 133 120 Schritte, Algorithmus 2 96 bis 121 (4 Bäume)" in text
    assert "d = 6 und Tiefe 8 braucht Algorithmus 2 in 8 von 8 Bäumen mehr (6 084 bis 9 509 gegen 5 074 bis 5 828; Erstmessung, in der Zweitmessung mit anderen Zufalls-Seeds in 6 von 8)" in text and "nicht gleich teuer" in text


def test_ensembles_verdikt_nennt_median_und_rang():
    zeilen = [{"modell": "Boosting 200 Bäume, Tiefe 3", "median": (0.065, 0.076), "rang1": (0.013, 0.087)}]
    kind, text = ev.verdict_ensembles(zeilen)
    assert kind == "info" and "Boosting 200 Bäume, Tiefe 3: Median 6,5 bis 7,6 %, anderes Rang 1 in 1,3 bis 8,7 % der Zeilen" in text and "fünf Instanzmengen zu je 30 Zeilen" in text


# ----------------------------------------------------------------------------------------------- Experiment 4
def _k(median, gepaart=True, m=24, anteil=0.375, sing=0, rang=(0.99, 1.0), q90=0.04):
    return {"modell": "Boosting Tiefe 3", "gepaart": gepaart, "m": m, "auswertungen": m * 30, "auswertungen_aufzaehlung": 1920, "anteil": anteil, "median_je_menge": median, "n_mengen": 5,
            "zeilen_je_menge": 10, "ziehungen": 10, "q90": q90, "rang1_je_menge": list(rang), "singulaer": sing, "n_ziehungen": 50}


def test_kernel_verdikt_schwellen_exakt_info_warnung():
    kind, text = ev.verdict_kernel(_k([0.0] * 5, rang=(1.0, 1.0), q90=0.0))
    assert kind == "success" and "M = 24 Koalitionen = 720 von 1 920 Modellauswertungen der Aufzählung (37,5 %)" in text and "Fehler ist null bis auf Rundung" in text
    kind, text = ev.verdict_kernel(_k([0.005, 0.01, 0.02, 0.0, 0.05]))
    assert kind == "info" and "Medianfehler je Instanzmenge 0,000 bis 0,050 (5 Mengen zu 10 Zeilen, je 10 Ziehungen), 90 %-Quantil 0,040" in text and "in 99 bis 100 % der Läufe" in text
    assert ev.verdict_kernel(_k([0.05, 0.0501, 0.01, 0.0, 0.0]))[0] == "warning"                  # größter Medianfehler 0,0501 liegt über 0,05
    kind, text = ev.verdict_kernel(_k([0.03] * 5, gepaart=False, m=1280, anteil=1.25))
    assert kind == "warning" and "teurer als die Aufzählung" in text and "einfach" in text


def test_kernel_verdikt_meldet_singulaere_ziehungen():
    kind, text = ev.verdict_kernel(_k([0.02] * 5, sing=3))
    assert "3 von 50 Ziehungen waren singulär" in text


# ----------------------------------------------------------------------------------------------- Experiment 5
def test_ordnung_verdikt_nennt_ordnung_und_fehler_je_tiefe():
    zeilen = [{"modell": f"Boosting Tiefe {k}", "ordnung": k} for k in (1, 2, 3, 4)]
    fehler = {"Boosting Tiefe 1": (1e-13, 1e-12), "Boosting Tiefe 2": (5e-13, 1e-12), "Boosting Tiefe 3": (0.005, 0.0204), "Boosting Tiefe 4": (0.013, 0.0426)}
    kind, text = ev.verdict_ordnung(zeilen, fehler)
    assert kind == "info"
    assert "Tiefe 1: Ordnung 1, Fehler 0,000; Tiefe 2: Ordnung 2, Fehler 0,000; Tiefe 3: Ordnung 3, Fehler 0,005 bis 0,020; Tiefe 4: Ordnung 4, Fehler 0,013 bis 0,043" in text
    assert "24 von 64 Koalitionen" in text and "kein Gesetz" in text


def _zs(d, o, m, anteile, sing=0):
    return {"d": d, "ordnung": o, "m": m, "n_spiele": 40, "anteile": anteile, "singulaer": sing, "ziehungen": 200}


def test_zufallsspiele_verdikt_zweige():
    kind, text = ev.verdict_zufallsspiele(_zs(6, 2, 12, [1.0, 0.975, 0.95, 1.0, 1.0]))
    assert kind == "success" and "Ordnung 2, d = 6, 12 gezogene Paare (24 Koalitionen): exakt in 95 bis 100 % von je 40 Spielen (5 Gruppen, zusammen 197 von 200 Spielen)" in text
    assert ev.verdict_zufallsspiele(_zs(6, 2, 8, [0.8, 0.85, 0.8, 0.8, 0.8]))[0] == "info"
    kind, text = ev.verdict_zufallsspiele(_zs(6, 3, 16, [0.0] * 5))
    assert kind == "warning" and "In keinem dieser Spiele exakt" in text
    kind, text = ev.verdict_zufallsspiele(_zs(6, 3, 16, [0.0, 0.025, 0.0, 0.0, 0.0]))
    assert kind == "warning" and "nicht allgemein exakt" in text and "In keinem" not in text
    kind, text = ev.verdict_zufallsspiele(_zs(10, 2, 8, [0.0] * 5, sing=200))
    assert kind == "warning" and "weniger als d − 1 = 9" in text and "200 von 200 Ziehungen singulär" in text
