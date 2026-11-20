"""Experimente 4 und 5 („KernelSHAP: Kosten gegen Fehler“, „Die Ordnung entscheidet“): friert die **Spieltabellen** ein, `data/shn_games_d6.json` und `data/shn_games_d10.json`.

Aufruf (Dev-venv, `OMP_NUM_THREADS=1`):
    python tools/shn_make_games.py             schreibt beide Dateien neu
    python tools/shn_make_games.py --zweit     Zweitmessung (andere Zufalls-Seeds) nach `data/shn_games_d6_zweit.json`, `..._d10_zweit.json`
    python tools/shn_make_games.py --check     rechnet neu und vergleicht (Band)

Ein **Spiel** einer Zeile x: v(S) für alle 2^d Koalitionen S, interventional (Mittel über den Hintergrund von 30 Trainingszeilen von f(x_S, Hintergrund_Rest)).
Die Tabelle ist das Spiel; die Aufzählung (`shapley_np`) gibt daraus den exakten Shapley-Wert, KernelSHAP zieht daraus Koalitionen. Die Modellauswertungen stecken im Freeze,
nicht in der App.
d = 6: die sechs Merkmale des Vehikels (Art, Kunde, Zoll, Wochenende, Vorlauf, Gewicht); Modelle auf n = 1000 Zeilen: Boosting (100 Bäume, Tiefe 1 bis 4), Wald (60 Bäume; Tiefe 2
und unbegrenzt), Netz (16 Einheiten). Fünf Modell-Zufalls-Seeds, je Seed eine **eigene Instanzmenge** (10 natürliche Zeilen), also fünf getrennte Instanzmengen.
d = 10: die sechs Merkmale plus vier unabhängige Rauschspalten N(0, 1) (Nullspieler; Modelle auf allen zehn Spalten); Boosting Tiefe 3, Wald, Netz; fünf Instanzmengen zu je 6 Zeilen."""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import shn_make_common as MC  # noqa: E402

import numpy as np  # noqa: E402

import shn_generator as G  # noqa: E402
from shn_make_learn import vergleiche  # noqa: E402
from shn_moebius import anteile_je_ordnung, moebius, ordnung  # noqa: E402

D6 = MC.DATA / "shn_games_d6.json"
D10 = MC.DATA / "shn_games_d10.json"
HINTERGRUND = 30

MODELLE6 = {
    "Boosting Tiefe 1": ("gbm", dict(n_estimators=100, max_depth=1)),
    "Boosting Tiefe 2": ("gbm", dict(n_estimators=100, max_depth=2)),
    "Boosting Tiefe 3": ("gbm", dict(n_estimators=100, max_depth=3)),
    "Boosting Tiefe 4": ("gbm", dict(n_estimators=100, max_depth=4)),
    "Wald Tiefe 2": ("rf", dict(max_depth=2)),
    "Wald unbegrenzt": ("rf", dict()),
    "Netz": ("mlp", dict()),
}
MODELLE10 = {
    "Boosting Tiefe 3": ("gbm", dict(n_estimators=100, max_depth=3)),
    "Wald unbegrenzt": ("rf", dict()),
    "Netz": ("mlp", dict()),
}
_MASKEN = {}


def maskentabelle(d):
    if d not in _MASKEN:
        _MASKEN[d] = np.array([[(k >> j) & 1 for j in range(d)] for k in range(1 << d)], dtype=bool)
    return _MASKEN[d]


def spiele(f, Xe, B, chunk_rows=120_000):
    """Interventionale Koalitionswerte (nx, 2^d): Mittel über den Hintergrund B von f(x_S, B_rest)."""
    d = Xe.shape[1]
    masken = maskentabelle(d)
    K, nb = 1 << d, len(B)
    step = max(1, chunk_rows // (K * nb))
    out = np.zeros((len(Xe), K))
    for a in range(0, len(Xe), step):
        xs = Xe[a:a + step]
        H = np.where(masken[None, :, None, :], xs[:, None, None, :], B[None, None, :, :])
        out[a:a + step] = f(H.reshape(-1, d)).reshape(len(xs), K, nb).mean(axis=2)
    return out


def daten_satz(sd, d, nx):
    X, y = G.daten(1000, sd)
    Xt, _ = G.daten(100, 5000 + sd)
    if d == 10:
        r = np.random.default_rng(77 + sd)
        X = np.column_stack([X, r.normal(size=(len(X), 4))])
        Xt = np.column_stack([Xt, r.normal(size=(len(Xt), 4))])
    B = X[np.random.default_rng(1).choice(len(X), HINTERGRUND, replace=False)]
    return X, y, Xt[:nx], B


def rechne_d(d, zweit=False):
    modelle = MODELLE6 if d == 6 else MODELLE10
    nx = 10 if d == 6 else 6
    seeds = MC.MODELL_SEEDS_ZWEIT if zweit else MC.MODELL_SEEDS
    out = {"d": d, "hintergrund": HINTERGRUND, "instanzen_je_menge": nx, "modell_seeds": list(seeds),
           "merkmale": list(G.NAMEN) + (["Rauschen 1", "Rauschen 2", "Rauschen 3", "Rauschen 4"] if d == 10 else []), "modelle": []}
    for nm, (art, kw) in modelle.items():
        kw = dict(kw)
        if art == "mlp":
            kw["nfeat"] = d
        mengen, r2s = [], []
        for sd in seeds:
            X, y, Xt, B = daten_satz(sd, d, nx)
            m = MC.mache_modell(art, sd, **kw).fit(X, y)
            mengen.append(spiele(m.predict, Xt, B))
            Xr, _ = G.daten(3000, 900 + sd)
            if d == 10:
                Xr = np.column_stack([Xr, np.random.default_rng(900 + sd).normal(size=(len(Xr), 4))])
            r2s.append(MC.r2(m.predict(Xr), G.wahrheit(Xr)))
        V = np.array(mengen)                                  # (Mengen, nx, 2^d)
        eintrag = {"modell": nm, "r2_gegen_mu": float(np.mean(r2s)), "v": V}
        if d == 6:
            c = moebius(V, d)
            eintrag["ordnung_max"] = int(ordnung(c, d, 1e-9))
            eintrag["anteile_je_ordnung"] = anteile_je_ordnung(c, d).tolist()
        out["modelle"].append(eintrag)
        print(f"d={d} {nm}: R2 gegen mu {np.mean(r2s):.3f}" + (f", Ordnung {eintrag['ordnung_max']}" if d == 6 else ""), flush=True)
    return out


def schreibe(zweit=False):
    for d, datei in ((6, D6), (10, D10)):
        if zweit:
            datei = datei.with_name(datei.stem + "_zweit.json")
        MC.schreibe_json(datei, rechne_d(d, zweit), "Erfundener Generator (verweilzeit-demo, kopiert); interventionale Spiele von scikit-learn-Modellen, Hintergrund 30 Trainingszeilen", stellen=12)
        print("geschrieben:", datei)


def pruefe(zweit=False):
    ok = True
    for d, datei in ((6, D6), (10, D10)):
        if zweit:
            datei = datei.with_name(datei.stem + "_zweit.json")
        neu = _tolist(rechne_d(d, zweit))
        fe = vergleiche(neu, MC.lade_json(datei), tol=1e-6)
        print(f"CHECK d={d}:", "gleich" if not fe else f"{len(fe)} Abweichungen, erste: {fe[:3]}")
        ok &= not fe
    return ok


def _tolist(x):
    if isinstance(x, dict):
        return {k: _tolist(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_tolist(v) for v in x]
    if isinstance(x, np.ndarray):
        return x.tolist()
    return x


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--zweit", action="store_true")
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args()
    if a.check:
        sys.exit(0 if pruefe(a.zweit) else 1)
    schreibe(a.zweit)
