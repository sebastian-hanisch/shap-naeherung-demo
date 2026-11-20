"""Die Werkzeuge `tools/shn_make_*.py` und die eingefrorenen Dateien: Neu-Rechnen als **Band** (Plattform: scikit-learn und BLAS können in der letzten Stelle abweichen,
Baumstrukturen bei Gleichständen; darum Toleranz bzw. Spanne, kein bitgleicher Vergleich). Ganze Läufe (`--check`) sind zu teuer für die Suite und stehen in der README."""

import sys
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("sklearn")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools"))
import shn_make_costs as MK  # noqa: E402
import shn_make_games as MG  # noqa: E402
import shn_make_learn as ML  # noqa: E402
import shn_make_common as MC  # noqa: E402

import shn_data as DATA  # noqa: E402
import shn_generator as G  # noqa: E402


def test_spieltabelle_eines_modells_neu_gerechnet_im_band():
    """Boosting Tiefe 1, Instanzmenge 1 (d = 6): neu trainiert und neu aufgezählt gleich der eingefrorenen Tabelle (relativ 1e-6)."""
    eintrag = next(m for m in DATA.games(6)["modelle"] if m["modell"] == "Boosting Tiefe 1")
    sd = DATA.games(6)["modell_seeds"][0]
    X, y, Xt, B = MG.daten_satz(sd, 6, 10)
    m = MC.mache_modell("gbm", sd, n_estimators=100, max_depth=1).fit(X, y)
    V = MG.spiele(m.predict, Xt, B)
    assert V == pytest.approx(np.array(eintrag["v"])[0], rel=1e-6, abs=1e-8)


def test_kostenzelle_neu_gezaehlt_liegt_in_der_eingefrorenen_spanne():
    z = next(c for c in DATA.costs()["tabelle"] if (c["d"], c["tiefe"]) == (6, 3))
    for sd in range(3):
        b, x = MK.baum_und_punkt(6, 3, sd, 77)
        v1, v2, dev = MK.zaehlen(b, x, 6)
        assert (v1, v2) == (z["aufzaehlung"][sd], z["alg2"][sd]) and dev < 1e-8


def test_toy_modell_beitraege_gleich_den_eingefrorenen_im_band():
    """Gieriger Baum, Tiefe 3, ohne Rauschen, n = 400, die ersten drei Zufalls-Seeds: pfadabhängig und interventional gleich 0 für x₂ (wie eingefroren)."""
    zeile = next(z for z in DATA.learn()["toy"]["zeilen"] if z["lernverfahren"].startswith("gieriger") and z["n"] == 400 and z["rauschen"] == 0.0)
    for sd in range(3):
        rng = np.random.default_rng([400, sd, 11])
        idx = rng.choice(8, size=400, p=ML.PR)
        X = ML.P8[idx]
        m = MC.mache_modell("baum", sd, max_depth=3).fit(X, X[:, 0])
        p, i, k, ph2, abw = ML.toy_modell_beitraege(m)
        assert p[1] == pytest.approx(zeile["phi2_je_seed"][sd], abs=1e-9) and i[1] == pytest.approx(0.0, abs=1e-9) and abw < 1e-12
        assert k == pytest.approx(zeile["kond"], abs=0.02)                                 # konditional des Modells: Mittel (0,34 / 0,20 / 0) bis auf Streuung zwischen Seeds


def test_wahrheit_vorlauf_geschlossen_gegen_stichprobe_im_band():
    """Orakel für die Wahrheit von g = Vorlauf: die geschlossenen Erwartungen gegen eine große Stichprobe aus dem Generator (Band mit Standardfehler)."""
    X, _ = G.daten(200000, 77)
    mu0 = MC.mittel_vorlauf()
    assert X[:, 4].mean() == pytest.approx(mu0, abs=4 * X[:, 4].std() / np.sqrt(len(X)))
    sel = X[:, 2] == 0                                                                    # bedingt auf Zoll = 0 (Art unbekannt)
    soll = (G.ART_P[1] * 5.5 + G.ART_P[2] * 3.25) / (G.ART_P[0] * (1 - G.P_ZOLL) + G.ART_P[1] + G.ART_P[2] + G.ART_P[3])
    assert X[sel, 4].mean() == pytest.approx(soll, abs=4 * X[sel, 4].std() / np.sqrt(sel.sum()))
    wi, wk = MC.wahrheit_vorlauf(np.array([[3.0, 0, 0, 0, 5.0, 0]]))                       # eine Zeile: Vorlauf 5, Art Umschlag
    assert wi[0, 4] == pytest.approx(5.0 - mu0) and wi[0, :4] == pytest.approx(0.0)       # interventional: nur der Vorlauf
    assert wk.shape == (1, 5) and wk.sum() == pytest.approx(5.0 - mu0)                     # Effizienz auch konditional


def test_schreibe_json_rundet_und_behaelt_winzige_werte(tmp_path):
    p = tmp_path / "x.json"
    MC.schreibe_json(p, {"a": 0.123456789, "b": 3e-14, "c": [np.float64(1.5), 2]}, "Test")
    d = MC.lade_json(p)
    assert d["a"] == 0.123457 and d["b"] == 3e-14 and d["c"] == [1.5, 2] and d["meta"]["quelle"] == "Test" and d["meta"]["stand"]
