"""Experiment 2 („Was macht ein Lernverfahren daraus?“): friert die Beiträge gelernter Modelle ein (`data/shn_learn.json`).

Aufruf (Dev-venv, `OMP_NUM_THREADS=1`):
    python tools/shn_make_learn.py             schreibt die Datei neu (Erstmessung; dauert einige Minuten)
    python tools/shn_make_learn.py --zweit     Zweitmessung mit anderen Zufalls-Seeds nach `data/shn_learn_zweit.json`
    python tools/shn_make_learn.py --check     rechnet neu und vergleicht mit der Datei (Gleitkomma-Band 1e-6 relativ)

**Teil A (Toy, vollständig aufgezählt):** Ziel f = x1 auf der binären Toy-Verteilung (x2 = x1 mit 9/10, x3 unabhängig). Zähleinheit: Zufalls-Seeds = Trainingsdatensätze (30 je Zelle).
Je Modell: Wichtigkeit (Mittel von |φ|, gewichtet mit der Verteilung) pfadabhängig (Blattanteile = Trainingsmasse), interventional **des Modells** und konditional **des Modells**,
alle drei exakt auf den acht Punkten; dazu der Anteil der Zufalls-Seeds mit pfadabhängigem |φ2| > 0,02.
**Teil B (Vehikel):** Ziel g = Vorlauf (liest nur den Vorlauf), n = 1000 Zeilen, fünf Merkmale, 5 Zufalls-Seeds, **fünf getrennte Instanzmengen zu je 25 Zeilen**; Hintergrund = die exakte
Bevölkerung des Generators (kein Stichprobenhintergrund); Wahrheit von g je Funktional geschlossen (`wahrheit_vorlauf`)."""

import argparse
import itertools
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import shn_make_common as MC  # noqa: E402

import numpy as np  # noqa: E402

import shn_generator as G  # noqa: E402
from shn_shapley import shapley_np  # noqa: E402
from shn_toy import PUNKTE, verteilung  # noqa: E402

DATEI = MC.DATA / "shn_learn.json"
DATEI_ZWEIT = MC.DATA / "shn_learn_zweit.json"
PR = np.array([float(verteilung(0.9)[p]) for p in PUNKTE])
P8 = np.array(PUNKTE, float)
CODE = np.array([4, 2, 1])
MASKEN3 = np.array([[(k >> j) & 1 for j in range(3)] for k in range(8)], dtype=bool)

TOY_LERNER = {
    "gieriger Baum, Tiefe 3": ("baum", dict(max_depth=3), dict(max_depth=3, min_samples_leaf=5)),
    "Zufallsbaum (ein zufälliges Merkmal, zufälliger Schnitt)": ("zufallsbaum", dict(max_features=1), dict(max_features=1, min_samples_leaf=5)),
    "Wald 60 Bäume, 1 Merkmal je Schnitt": ("rf", dict(n_estimators=60, max_features=1, min_samples_leaf=1), dict(n_estimators=60, max_features=1, min_samples_leaf=5)),
    "Wald 60 Bäume, 2 Merkmale je Schnitt": ("rf", dict(n_estimators=60, max_features=2, min_samples_leaf=1), dict(n_estimators=60, max_features=2, min_samples_leaf=5)),
    "Boosting 100 Bäume, Tiefe 3": ("gbm", dict(n_estimators=100, max_depth=3), dict(n_estimators=100, max_depth=3, min_samples_leaf=5)),
}
VEH_LERNER = {
    "gieriger Baum, Tiefe 6": ("baum", dict(max_depth=6, min_samples_leaf=5)),
    "Wald 60 Bäume, alle Merkmale je Schnitt": ("rf", dict(max_features=1.0)),
    "Wald 60 Bäume, 0,5 der Merkmale je Schnitt": ("rf", dict(max_features=0.5)),
    "Wald 60 Bäume, 1 Merkmal je Schnitt": ("rf", dict(max_features=1)),
    "Boosting 200 Bäume, Tiefe 3": ("gbm", dict(n_estimators=200, max_depth=3)),
}


def toy_modell_beitraege(m):
    """Wichtigkeit (3 Werte) pfadabhängig / interventional des Modells / konditional des Modells, exakt auf den acht Punkten."""
    tab = m.predict(P8)
    V = MC.pfad_werte(m, P8, 3)                    # (8 Punkte, 8 Koalitionen), v(alle) = Vorhersage ist geprüft
    pfad = shapley_np(V, 3)
    vint = np.zeros((8, 8))
    vkon = np.zeros((8, 8))
    for xi, x in enumerate(P8):
        for k in range(8):
            Z = np.where(MASKEN3[k][None, :], x[None, :], P8)
            vint[xi, k] = PR @ tab[(Z @ CODE).astype(int)]
            sel = np.all(P8[:, MASKEN3[k]] == x[MASKEN3[k]][None, :], axis=1)
            vkon[xi, k] = (PR[sel] @ tab[sel]) / PR[sel].sum()
    assert np.abs(pfad.sum(axis=1) - (tab - V[:, 0])).max() < 1e-9, "Effizienz"
    return [PR @ np.abs(shapley_np(v, 3)) for v in (V, vint, vkon)] + [pfad[:, 1], float(np.abs(tab - P8[:, 0]).max())]


def teil_a(zweit=False):
    zeilen = []
    seeds = range(30) if not zweit else range(100, 130)
    for nm, (art, kw0, kw1) in TOY_LERNER.items():
        for n in (400, 4000):
            for rausch in (0.0, 0.5):
                if rausch == 0.5 and art in ("zufallsbaum",):
                    continue
                kw = kw0 if rausch == 0.0 else kw1
                acc = {"pfad": [], "interv": [], "kond": []}
                phi2 = []
                abw = 0.0
                for sd in seeds:
                    rng = np.random.default_rng([n, sd, 11])
                    idx = rng.choice(8, size=n, p=PR)
                    X = P8[idx]
                    y = X[:, 0] + rausch * rng.standard_normal(n)
                    m = MC.mache_modell(art, sd, **kw).fit(X, y)
                    p, i, k, ph2, ab = toy_modell_beitraege(m)
                    abw = max(abw, ab)
                    acc["pfad"].append(p)
                    acc["interv"].append(i)
                    acc["kond"].append(k)
                    phi2.append(float(p[1]))
                mean = {k: np.mean(v, axis=0).tolist() for k, v in acc.items()}
                zeilen.append({"lernverfahren": nm, "n": n, "rauschen": rausch, "pfad": mean["pfad"], "interv": mean["interv"], "kond": mean["kond"],
                               "anteil_phi2": float(np.mean(np.array(phi2) > 0.02)), "phi2_je_seed": phi2, "n_seeds": len(phi2),
                               "abweichung_von_f": abw})
                print(f"toy {nm} n={n} rauschen={rausch}: pfad {np.round(mean['pfad'], 3)} interv {np.round(mean['interv'], 3)} Anteil {zeilen[-1]['anteil_phi2']:.2f}", flush=True)
    # Wahrheit der Funktion f = x1 (kein Modell)
    tab = P8[:, 0]
    wi = np.zeros((8, 8))
    wk = np.zeros((8, 8))
    for xi, x in enumerate(P8):
        for k in range(8):
            Z = np.where(MASKEN3[k][None, :], x[None, :], P8)
            wi[xi, k] = PR @ tab[(Z @ CODE).astype(int)]
            sel = np.all(P8[:, MASKEN3[k]] == x[MASKEN3[k]][None, :], axis=1)
            wk[xi, k] = (PR[sel] @ tab[sel]) / PR[sel].sum()
    return {"q": 0.9, "zeilen": zeilen,
            "wahrheit": {"interventional": (PR @ np.abs(shapley_np(wi, 3))).tolist(), "konditional": (PR @ np.abs(shapley_np(wk, 3))).tolist()}}


def teil_b(zweit=False):
    mengen = MC.MENGEN_ZWEIT if zweit else MC.MENGEN
    seeds = MC.MODELL_SEEDS_ZWEIT if zweit else MC.MODELL_SEEDS
    Xp, wp = MC.pop_punkte(12)
    sets = [G.daten(25, s)[0] for s in mengen]
    wahr_i, wahr_k = [], []
    for Xe in sets:
        wi, wk = MC.wahrheit_vorlauf(Xe)
        wahr_i.append(np.abs(wi).mean(axis=0).tolist())
        wahr_k.append(np.abs(wk).mean(axis=0).tolist())
    zeilen = []
    for nm, (art, kw) in VEH_LERNER.items():
        pf = [[] for _ in sets]
        iv = [[] for _ in sets]
        r2s = []
        for sd in seeds:
            X, _ = G.daten(1000, sd)
            Xt, _ = G.daten(3000, 900 + sd)
            m = MC.mache_modell(art, sd, **kw).fit(X[:, :5], X[:, 4])
            f = lambda Z, m=m: m.predict(Z[:, :5])
            r2s.append(MC.r2(f(Xt), Xt[:, 4]))
            for k, Xe in enumerate(sets):
                pf[k].append(np.abs(shapley_np(MC.pfad_werte(m, Xe[:, :5], 5), 5)).mean(axis=0))
                iv[k].append(np.abs(shapley_np(MC.vals_pop(f, Xe, Xp, wp), 5)).mean(axis=0))
        zeilen.append({"lernverfahren": nm, "r2": float(np.mean(r2s)), "pfad_je_menge": [np.mean(a, axis=0).tolist() for a in pf],
                       "interv_je_menge": [np.mean(a, axis=0).tolist() for a in iv]})
        a_i = [z[0] for z in zeilen[-1]["interv_je_menge"]]
        a_p = [z[0] for z in zeilen[-1]["pfad_je_menge"]]
        print(f"vehikel {nm}: R2 {np.mean(r2s):.4f}; |phi_Art| interv {min(a_i):.3f}..{max(a_i):.3f}; pfad {min(a_p):.3f}..{max(a_p):.3f}", flush=True)
    return {"mengen": list(mengen), "modell_seeds": list(seeds), "n_je_menge": 25, "n_train": 1000, "merkmale": list(G.NAMEN[:5]), "zeilen": zeilen,
            "wahrheit_je_menge": {"interventional": wahr_i, "konditional": wahr_k}}


def rechne(zweit=False):
    return {"toy": teil_a(zweit), "vehikel": teil_b(zweit)}


def vergleiche(neu, alt, pfad="", tol=1e-6, fehler=None):
    fehler = [] if fehler is None else fehler
    if isinstance(alt, dict):
        for k in alt:
            if k == "meta":
                continue
            vergleiche(neu[k], alt[k], pfad + "/" + k, tol, fehler)
    elif isinstance(alt, list):
        if len(alt) != len(neu):
            fehler.append((pfad, "Länge"))
        for i, (a, b) in enumerate(zip(alt, neu)):
            vergleiche(b, a, f"{pfad}[{i}]", tol, fehler)
    elif isinstance(alt, float):
        if abs(neu - alt) > tol * max(1.0, abs(alt)) + 2e-6:
            fehler.append((pfad, neu, alt))
    elif neu != alt:
        fehler.append((pfad, neu, alt))
    return fehler


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--zweit", action="store_true")
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args()
    out = rechne(a.zweit)
    if a.check:
        alt = MC.lade_json(DATEI_ZWEIT if a.zweit else DATEI)
        fe = vergleiche(MC.json.loads(MC.json.dumps(out)), alt)
        print("CHECK:", "gleich" if not fe else f"{len(fe)} Abweichungen, erste: {fe[:3]}")
        sys.exit(1 if fe else 0)
    MC.schreibe_json(DATEI_ZWEIT if a.zweit else DATEI, out, "Erfundener Generator (verweilzeit-demo, kopiert) und binäres Toy; scikit-learn-Modelle, eigene Rechenwege (shn_tree)")
    print("geschrieben:", DATEI_ZWEIT if a.zweit else DATEI)
