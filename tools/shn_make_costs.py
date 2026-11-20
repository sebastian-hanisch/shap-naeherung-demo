"""Experiment 3 („Baum-SHAP: zwei Rechenwege, Kosten und Abstand“): friert die Kostentabelle, den Abstand pfadabhängig gegen interventional am Ensemble und die kleinen
Bäume für den Live-Teil ein (`data/shn_costs.json`).

Aufruf (Dev-venv, `OMP_NUM_THREADS=1`):
    python tools/shn_make_costs.py             schreibt die Datei neu (dauert gut zehn Minuten, vor allem der Ensemble-Abstand)
    python tools/shn_make_costs.py --zweit     Zweitmessung (andere Zufalls-Seeds) nach `data/shn_costs_zweit.json`
    python tools/shn_make_costs.py --check     rechnet neu und vergleicht (ganze Zahlen exakt, Gleitkomma im Band)

**Zählregeln:** Rechenweg 1 (Aufzählung) zählt Besuche der Knotenrekursion EXPVALUE über alle 2^d Koalitionen; Rechenweg 2 (Algorithmus 2) zählt je Aufruf der Knotenrekursion einen
Schritt plus je eine Schleifenrunde in EXTEND und UNWIND. Die zwei Einheiten sind nicht gleich teuer: die Tabelle zeigt das Wachstum, nicht den Preis.
**Zählzellen:** acht Bäume je Zelle (vier bei d = 14), je Baum eine Instanz; Trainingsdaten eine erfundene Zielfunktion über d gleichverteilten Merkmalen, n = 3000 (Zufalls-Seed = Baumnummer)."""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import shn_make_common as MC  # noqa: E402

import numpy as np  # noqa: E402

import shn_generator as G  # noqa: E402
from shn_shapley import shapley_np  # noqa: E402
from shn_tree import Baum, werte_aufzaehlung  # noqa: E402
from shn_treeshap import besuche_vollstaendiger_baum, schritte_vollstaendiger_baum, treeshap_poly  # noqa: E402

DATEI = MC.DATA / "shn_costs.json"
DATEI_ZWEIT = MC.DATA / "shn_costs_zweit.json"
ZELLEN = [(d, t) for d in (6, 8, 10, 12, 14) for t in (3, 6, 8)]


def ziel(X, coef):
    a, bq = coef
    d = X.shape[1]
    y = np.zeros(len(X))
    for j in range(d):
        y += a[j] * np.sin(3 * X[:, j] + j)
    for j in range(d - 1):
        y += bq[j] * X[:, j] * X[:, j + 1]
    return y


def wiederholt(b: Baum) -> float:
    """Anteil der Blätter, deren Pfad ein Merkmal mehr als einmal enthält."""
    rep = tot = 0
    stack = [(0, [])]
    while stack:
        i, pfad = stack.pop()
        if b.left[i] == -1:
            tot += 1
            rep += len(pfad) != len(set(pfad))
            continue
        for c in (b.left[i], b.right[i]):
            stack.append((c, pfad + [b.feat[i]]))
    return rep / tot


def baum_und_punkt(d, tiefe, sd, salt):
    rng = np.random.default_rng([d, tiefe, sd, salt])
    coef = (rng.normal(size=d), rng.normal(size=d - 1))
    X = rng.uniform(-1, 1, size=(3000, d))
    y = ziel(X, coef) + 0.2 * rng.standard_normal(3000)
    t = MC.DecisionTreeRegressor(max_depth=tiefe, min_samples_leaf=3, random_state=sd).fit(X, y)
    x = rng.uniform(-1, 1, size=d)
    return MC.baum_aus_sklearn(t), x


def zaehlen(b: Baum, x, d):
    vals, besuche = werte_aufzaehlung(b, x, d)
    phi1 = shapley_np(np.array(vals)[None, :], d)[0]
    phi2, schritte = treeshap_poly(b, x, d)
    return besuche, schritte, float(np.abs(phi1 - np.array(phi2)).max())


def voll_baum(d, ebene=0):
    if ebene == d:
        return ("L", float(ebene))
    k = 2 ** (d - ebene - 1)
    return ("N", ebene, 0, voll_baum(d, ebene + 1), voll_baum(d, ebene + 1), float(k), float(k))


def tabelle(zweit=False):
    salt = 78 if zweit else 77
    zeilen = []
    for d, tiefe in ZELLEN:
        n_baeume = 8 if d < 14 else 4
        bl, c1, c2, rep, dev = [], [], [], [], 0.0
        for sd in range(n_baeume):
            b, x = baum_und_punkt(d, tiefe, sd, salt)
            v1, v2, de = zaehlen(b, x, d)
            assert de < 1e-8, "Algorithmus 2 weicht von der Aufzählung ab"
            bl.append(b.n_blaetter())
            rep.append(wiederholt(b))
            c1.append(v1)
            c2.append(v2)
            dev = max(dev, de)
        zeilen.append({"d": d, "tiefe": tiefe, "n_baeume": n_baeume, "blaetter": bl, "aufzaehlung": c1, "alg2": c2, "wiederholt": rep, "abweichung_max": dev})
        r = np.array(c1) / np.array(c2)
        print(f"d={d} Tiefe={tiefe}: Aufzählung {int(np.mean(c1))}, Algorithmus 2 {int(np.mean(c2))}, Verhältnis {r.min():.2f}..{r.max():.2f}, Alg. 2 mehr in {int((r < 1).sum())}/{len(r)}", flush=True)
    voll = []
    for d in range(1, 9):
        b = Baum.aus_tupel(voll_baum(d))
        v1, v2, de = zaehlen(b, (1.0,) * d, d)
        assert v1 == besuche_vollstaendiger_baum(d) and v2 == schritte_vollstaendiger_baum(d) and de < 1e-9
        voll.append({"d": d, "aufzaehlung": v1, "alg2": v2})
    return zeilen, voll


def kleine_baeume():
    out = []
    for d in range(4, 11):
        for tiefe in (3, 6, 8):
            b, x = baum_und_punkt(d, tiefe, 0, 99)
            v1, v2, _ = zaehlen(b, x, d)
            out.append({"d": d, "tiefe": tiefe, "baum": b.als_dict(), "x": [float(e) for e in x], "besuche": v1, "schritte": v2, "blaetter": b.n_blaetter()})
    return out


def ensembles(zweit=False):
    mengen = MC.MENGEN_ZWEIT if zweit else MC.MENGEN
    seeds = MC.MODELL_SEEDS_ZWEIT if zweit else MC.MODELL_SEEDS
    Xp, wp = MC.pop_punkte(24)
    sets = [G.daten(30, s)[0] for s in mengen]
    modelle = {
        "Boosting 200 Bäume, Tiefe 3": ("gbm", dict(n_estimators=200, max_depth=3, learning_rate=0.1)),
        "Wald 60 Bäume, 0,5 der Merkmale je Schnitt": ("rf", dict(max_features=0.5)),
        "Wald 60 Bäume, alle Merkmale je Schnitt": ("rf", dict(max_features=1.0)),
    }
    res = []
    for nm, (art, kw) in modelle.items():
        rel = [[] for _ in sets]
        top = [[] for _ in sets]
        sg = [[] for _ in sets]
        r2s = []
        for sd in seeds:
            X, y = G.daten(1000, sd)
            Xt, _ = G.daten(3000, 900 + sd)
            m = MC.mache_modell(art, sd, **kw).fit(X[:, :5], y)
            f = lambda Z, m=m: m.predict(Z[:, :5])
            r2s.append(MC.r2(f(Xt), G.wahrheit(Xt)))
            for k, Xe in enumerate(sets):
                pp = shapley_np(MC.pfad_werte(m, Xe[:, :5], 5), 5)
                pi = shapley_np(MC.vals_pop(f, Xe, Xp, wp), 5)
                rel[k].append(np.abs(pp - pi).sum(axis=1) / np.abs(pi).sum(axis=1))
                top[k].append(np.argmax(np.abs(pp), axis=1) != np.argmax(np.abs(pi), axis=1))
                gro = np.argmax(np.abs(pi), axis=1)
                ar = np.arange(len(Xe))
                sg[k].append(np.sign(pp[ar, gro]) != np.sign(pi[ar, gro]))
        res.append({"modell": nm, "r2_gegen_mu": float(np.mean(r2s)),
                    "median_je_menge": [float(np.median(np.concatenate(r))) for r in rel],
                    "q90_je_menge": [float(np.quantile(np.concatenate(r), 0.9)) for r in rel],
                    "rang1_anders_je_menge": [float(np.concatenate(t).mean()) for t in top],
                    "vorzeichen_anders_je_menge": [float(np.concatenate(s).mean()) for s in sg]})
        print(f"ensemble {nm}: Median {np.round(res[-1]['median_je_menge'], 3)}, Rang 1 anders {np.round(res[-1]['rang1_anders_je_menge'], 3)}", flush=True)
    return {"mengen": list(mengen), "modell_seeds": list(seeds), "n_je_menge": 30, "zeilen": res}


def rechne(zweit=False, mit_ensembles=True):
    zeilen, voll = tabelle(zweit)
    out = {"tabelle": zeilen, "vollstaendig": voll, "kleine_baeume": kleine_baeume()}
    if mit_ensembles:
        out["ensembles"] = ensembles(zweit)
    return out


if __name__ == "__main__":
    from shn_make_learn import vergleiche

    ap = argparse.ArgumentParser()
    ap.add_argument("--zweit", action="store_true")
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--ohne-ensembles", action="store_true", help="nur für schnelle Läufe; die Datei bekommt dann keine Ensemble-Zeilen")
    a = ap.parse_args()
    out = rechne(a.zweit, not a.ohne_ensembles)
    if a.check:
        alt = MC.lade_json(DATEI_ZWEIT if a.zweit else DATEI)
        alt = {k: v for k, v in alt.items() if k in out}
        fe = vergleiche(MC.json.loads(MC.json.dumps(out)), alt)
        print("CHECK:", "gleich" if not fe else f"{len(fe)} Abweichungen, erste: {fe[:3]}")
        sys.exit(1 if fe else 0)
    MC.schreibe_json(DATEI_ZWEIT if a.zweit else DATEI, out, "Erfundene Zielfunktion über gleichverteilten Merkmalen (Kosten), Generator der Verweilzeit-Demo (Ensembles); scikit-learn-Bäume, eigene Rechenwege")
    print("geschrieben:", DATEI_ZWEIT if a.zweit else DATEI)
