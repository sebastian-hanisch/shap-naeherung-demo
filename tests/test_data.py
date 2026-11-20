"""Eingefrorene Messungen (`data/*.json`): Aufbau, Quelle und Stand je Datei, innere Widerspruchsfreiheit (der Test rechnet nur mit den Daten und den Rechenwegen der Demo)."""

import numpy as np
import pytest

import shn_data as DATA
import shn_experiments as E
from shn_shapley import shapley_np


@pytest.mark.parametrize("name", ["learn", "costs"])
def test_meta_quelle_und_stand_stehen_in_jeder_datei(name):
    for zweit in (False, True):
        meta = getattr(DATA, name)(zweit)["meta"]
        assert meta["quelle"] and len(meta["stand"]) == 10 and meta["scikit-learn"] and meta["numpy"]


def test_meta_der_spieltabellen():
    for d in (6, 10):
        for zweit in (False, True):
            g = DATA.games(d, zweit)
            assert g["meta"]["quelle"] and g["d"] == d and g["hintergrund"] == 30
            assert len(g["merkmale"]) == d


def test_lernzeilen_toy_und_vehikel_sind_vollstaendig():
    L = DATA.learn()
    assert len(L["toy"]["zeilen"]) == 18 and len(L["vehikel"]["zeilen"]) == 5
    for z in L["toy"]["zeilen"]:
        assert z["n_seeds"] == 30 and len(z["phi2_je_seed"]) == 30
        assert z["anteil_phi2"] == pytest.approx(np.mean(np.array(z["phi2_je_seed"]) > 0.02))
        assert z["pfad"][1] == pytest.approx(np.mean(z["phi2_je_seed"]), abs=1e-5)         # phi2_je_seed ist die Wichtigkeit von x2 je Zufalls-Seed, pfad[1] ihr Mittel
    assert L["toy"]["wahrheit"] == {"interventional": [0.5, 0.0, 0.0], "konditional": [0.34, 0.2, 0.0]}
    for z in L["vehikel"]["zeilen"]:
        assert len(z["pfad_je_menge"]) == len(z["interv_je_menge"]) == 5 and all(len(m) == 5 for m in z["pfad_je_menge"])
    assert L["vehikel"]["mengen"] == [4242, 6201, 6202, 6203, 6204] and L["vehikel"]["n_je_menge"] == 25


def test_wahrheit_vom_vorlauf_ist_interventional_nur_der_vorlauf():
    w = DATA.learn()["vehikel"]["wahrheit_je_menge"]
    assert all(m[j] == 0 for m in w["interventional"] for j in range(4)) and all(m[4] > 2.0 for m in w["interventional"])
    assert all(m[1] == 0 and m[3] == 0 and m[0] > 0.9 for m in w["konditional"])         # Kunde und Wochenende bleiben auch konditional Nullspieler


def test_kostentabelle_zellen_und_abweichungen():
    C = DATA.costs()
    assert [(z["d"], z["tiefe"]) for z in C["tabelle"]] == [(d, t) for d in (6, 8, 10, 12, 14) for t in (3, 6, 8)]
    for z in C["tabelle"]:
        assert z["n_baeume"] == (8 if z["d"] < 14 else 4) == len(z["aufzaehlung"]) == len(z["alg2"]) == len(z["blaetter"])
        assert z["abweichung_max"] < 1e-8                                                 # Algorithmus 2 gleich Aufzählung in jeder Zelle
    assert [(v["d"], v["aufzaehlung"], v["alg2"]) for v in C["vollstaendig"]][:3] == [(1, 5, 7), (2, 19, 33), (3, 65, 121)]


def test_kleine_baeume_zaehler_stimmen_mit_neu_gerechnetem_ueberein():
    """Die eingefrorenen Zähler (Besuche, Schritte) gleich denen, die die Demo hier an demselben Baum und Punkt neu zählt (alle 21 Bäume)."""
    kb = DATA.costs()["kleine_baeume"]
    assert sorted((e["d"], e["tiefe"]) for e in kb) == [(d, t) for d in range(4, 11) for t in (3, 6, 8)]
    for e in kb:
        r = E.exp3(f"trainiert, Tiefe {e['tiefe']}", e["d"])
        assert (r["besuche"], r["schritte"], r["blaetter"]) == (e["besuche"], e["schritte"], e["blaetter"])
        assert r["abweichung"] < 1e-8 and r["effizienz"] < 1e-8


@pytest.mark.parametrize("d", [6, 10])
def test_spieltabellen_aufbau_und_effizienz(d):
    g = DATA.games(d)
    for m in g["modelle"]:
        V = np.array(m["v"])
        assert V.shape == (5, g["instanzen_je_menge"], 1 << d)
        phi = shapley_np(V, d)
        assert np.abs(phi.sum(axis=-1) - (V[..., -1] - V[..., 0])).max() < 1e-9
        assert 0.4 < m["r2_gegen_mu"] <= 1.0                                              # R² gegen μ (Wald Tiefe 2 und Boosting Tiefe 1 sind schwache Modelle)


def test_ordnung_aus_den_tabellen_gleich_der_eingefrorenen():
    for z in E.ordnung_tabelle():
        assert z["ordnung"] == z["ordnung_eingefroren"]
        assert z["anteile"] == pytest.approx(z["anteile_eingefroren"], abs=1e-6)
        assert sum(z["anteile"]) == pytest.approx(1.0)


def test_zweitmessung_hat_dieselbe_struktur_und_andere_zufalls_seeds():
    a, b = DATA.games(6), DATA.games(6, True)
    assert [m["modell"] for m in a["modelle"]] == [m["modell"] for m in b["modelle"]] and a["modell_seeds"] != b["modell_seeds"]
    assert not np.allclose(a["modelle"][2]["v"], b["modelle"][2]["v"])
    assert DATA.learn()["vehikel"]["mengen"] != DATA.learn(True)["vehikel"]["mengen"]
