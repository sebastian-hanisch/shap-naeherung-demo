"""Gemeinsame Hilfen der Werkzeuge `shn_make_*.py` (Dev-venv mit scikit-learn; die App selbst braucht das nicht).

Hier stehen: Modellfabrik (scikit-learn-Modelle, Hyperparameter fest je Stück), Umwandlung der Bäume in `shn_tree.Baum`, die exakte Bevölkerung des
Generators als gewichtete Stützpunkte (interventionaler Hintergrund ohne Stichprobe), die geschlossene konditionale Wahrheit für das Ziel „Vorlauf“ und das Schreiben
der eingefrorenen JSON-Dateien mit Quelle und Stand."""

import datetime
import json
import os
import sys
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.dont_write_bytecode = True
os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np  # noqa: E402
from sklearn.compose import ColumnTransformer  # noqa: E402
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor  # noqa: E402
from sklearn.neural_network import MLPRegressor  # noqa: E402
from sklearn.pipeline import make_pipeline  # noqa: E402
from sklearn.preprocessing import OneHotEncoder, StandardScaler  # noqa: E402
from sklearn.tree import DecisionTreeRegressor, ExtraTreeRegressor  # noqa: E402

import shn_generator as G  # noqa: E402
from shn_shapley import shapley_np  # noqa: E402
from shn_tree import Baum, blatt_rechnung  # noqa: E402

warnings.filterwarnings("ignore")
DATA = ROOT / "data"

# Zufalls-Seeds der Instanzmengen: die erste Menge ist die der Vormessung (4242), die vier weiteren sind getrennte Mengen; die Zweitmessung nimmt andere.
MENGEN = (4242, 6201, 6202, 6203, 6204)
MENGEN_ZWEIT = (7301, 7302, 7303, 7304, 7305)
MODELL_SEEDS = (0, 1, 2, 3, 4)
MODELL_SEEDS_ZWEIT = (10, 11, 12, 13, 14)


def heute() -> str:
    return datetime.date.today().isoformat()


def schreibe_json(pfad: Path, inhalt: dict, quelle: str, stellen: int = 6):
    """Schreibt `inhalt` mit Meta-Block (Quelle, Stand, Versionen) und rundet Gleitkommazahlen auf `stellen` Nachkommastellen."""
    import sklearn

    def runde(x):
        if isinstance(x, float):
            if x != 0.0 and abs(x) < 10.0 ** -(stellen - 1):
                return float(f"{x:.3g}")                      # winzige Werte (Rundungsfehler, Abweichungen) behalten drei gültige Stellen statt zu 0 zu werden
            return round(x, stellen)
        if isinstance(x, dict):
            return {k: runde(v) for k, v in x.items()}
        if isinstance(x, (list, tuple)):
            return [runde(v) for v in x]
        if isinstance(x, np.ndarray):
            return runde(x.tolist())
        if isinstance(x, (np.floating,)):
            return runde(float(x))
        if isinstance(x, (np.integer,)):
            return int(x)
        return x

    out = {"meta": {"quelle": quelle, "stand": heute(), "scikit-learn": sklearn.__version__, "numpy": np.__version__,
                    "hinweis": "Eingefrorene Messung (Dev-venv); die App rechnet nur mit diesen Zahlen, Neu-Rechnen und Abgleich mit `--check`."}}
    out.update(runde(inhalt))
    pfad.parent.mkdir(parents=True, exist_ok=True)
    pfad.write_text(json.dumps(out, ensure_ascii=False, separators=(",", ":")) + "\n", encoding="utf-8")


def lade_json(pfad: Path) -> dict:
    return json.loads(pfad.read_text(encoding="utf-8"))


# ------------------------------------------------------------------------------------------------ Bäume und Modelle
def baum_aus_sklearn(tree) -> Baum:
    """`shn_tree.Baum` aus einem angepassten scikit-learn-Baum (Blattanteile = weighted_n_node_samples, bei Bootstrap die Gewichte)."""
    t = tree.tree_
    return Baum(t.children_left.tolist(), t.children_right.tolist(), t.feature.tolist(), t.threshold.tolist(),
                t.value[:, 0, 0].tolist(), t.weighted_n_node_samples.tolist())


def zerlegung(m):
    """Modell = Konstante + Σ Gewicht · Baum: (Liste der `Baum`, Liste der Gewichte, Konstante)."""
    if isinstance(m, GradientBoostingRegressor):
        return [baum_aus_sklearn(e[0]) for e in m.estimators_], [m.learning_rate] * len(m.estimators_), float(np.ravel(m.init_.constant_)[0])
    if hasattr(m, "estimators_"):
        return [baum_aus_sklearn(e) for e in m.estimators_], [1.0 / len(m.estimators_)] * len(m.estimators_), 0.0
    return [baum_aus_sklearn(m)], [1.0], 0.0


def pfad_werte(m, X, d: int):
    """Pfadabhängige Koalitionswerte des Baummodells (Rechenweg: Blattanteile, vektorisiert): (n, 2^d). Orakel `v(alle) = Vorhersage` wird geprüft."""
    ts, ws, c0 = zerlegung(m)
    V = c0 + sum(w * blatt_rechnung(t, X, d) for t, w in zip(ts, ws))
    pred = m.predict(X)
    assert np.abs(V[:, -1] - pred).max() < 1e-8, "v(alle) = Vorhersage verletzt"
    return V


def mache_modell(art: str, seed: int, **hp):
    if art == "gbm":
        p = dict(n_estimators=200, max_depth=3, learning_rate=0.1)
        p.update(hp)
        return GradientBoostingRegressor(random_state=seed, **p)
    if art == "rf":
        p = dict(n_estimators=60, min_samples_leaf=3, max_features=0.5, n_jobs=1)
        p.update(hp)
        return RandomForestRegressor(random_state=seed, **p)
    if art == "mlp":
        nfeat = hp.pop("nfeat", 6)
        p = dict(hidden_layer_sizes=(16,), alpha=3.0, max_iter=500, solver="lbfgs")
        p.update(hp)
        ct = ColumnTransformer([("oh", OneHotEncoder(handle_unknown="ignore", sparse_output=False), [0, 1]),
                                ("sk", StandardScaler(), [2, 3, 4, 5] + list(range(6, nfeat)))])
        return make_pipeline(ct, MLPRegressor(random_state=seed, **p))
    if art == "baum":
        return DecisionTreeRegressor(random_state=seed, **hp)
    if art == "zufallsbaum":
        return ExtraTreeRegressor(random_state=seed, **hp)
    raise ValueError(art)


def r2(pred, ziel) -> float:
    return 1.0 - float(np.sum((pred - ziel) ** 2) / np.sum((ziel - ziel.mean()) ** 2))


# ------------------------------------------------------------------------------------------------ Bevölkerung des Generators
def _knoten(lo, hi, q):
    h = (hi - lo) / q
    return lo + (np.arange(q) + 0.5) * h, np.full(q, 1.0 / q)


def pop_punkte(q: int = 12):
    """(X (N, 6), w (N,)): die gemeinsame Verteilung der fünf Merkmale des Generators als gewichtete Stützpunkte (Vorlauf mit q Mittelpunkten je Art), Gewicht-Spalte 0.
    Damit ist der interventionale Wert ein Wert der Verteilung, keiner Stichprobe."""
    X, w = [], []
    for a in range(4):
        for k in range(G.K_KUNDEN):
            zs = [(0, 1.0)] if a != 0 else [(0, 1 - G.P_ZOLL), (1, G.P_ZOLL)]
            for z, pz in zs:
                for we, pw in ((0, 5 / 7), (1, 2 / 7)):
                    if a == 1:
                        vs, wv = _knoten(1.0, 10.0, q)
                    elif a == 2:
                        vs, wv = _knoten(0.5, 6.0, q)
                    else:
                        vs, wv = np.array([0.0]), np.array([1.0])
                    for v, wvv in zip(vs, wv):
                        X.append([a, k, z, we, v, 0.0])
                        w.append(G.ART_P[a] / G.K_KUNDEN * pz * pw * wvv)
    X, w = np.array(X, float), np.array(w, float)
    assert abs(w.sum() - 1) < 1e-12
    return X, w


MASKEN5 = np.array([[(k >> j) & 1 for j in range(5)] for k in range(32)], dtype=bool)


def vals_pop(f, Xe, Xp, wp, chunk=400_000):
    """Interventionale Koalitionswerte (ne, 32) über die fünf Merkmale mit gewichtetem Hintergrund: v(S) = Σ w_b f(x_S, b_rest)."""
    ne, N = len(Xe), len(Xp)
    out = np.zeros((ne, 32))
    step = max(1, chunk // (32 * N))
    for a in range(0, ne, step):
        xs = Xe[a:a + step]
        H = np.where(MASKEN5[None, :, None, :], xs[:, None, None, :5], Xp[None, None, :, :5])
        Z = np.zeros(H.shape[:-1] + (6,))
        Z[..., :5] = H
        out[a:a + step] = f(Z.reshape(-1, 6)).reshape(len(xs), 32, N) @ wp
    return out


def mittel_vorlauf() -> float:
    """Erwartung des Vorlaufs unter der Verteilung des Generators: Export 5,5 (Anteil 0,30) und Umschlag 3,25 (Anteil 0,25)."""
    return G.ART_P[1] * 5.5 + G.ART_P[2] * 3.25


def wahrheit_vorlauf(Xe):
    """Wahrheit je Funktional für g = Vorlauf (Spalte 4), geschlossen aus der Verteilung (kein Modell, keine Stichprobe): (interventional (n,5), konditional (n,5)).
    Interventional: nur der Vorlauf bekommt etwas (x − E[Vorlauf]). Konditional: v(S) = x4 mit Vorlauf in S, sonst E[Vorlauf | Art und Zoll, soweit in S]."""
    mu0 = mittel_vorlauf()
    mittel_art = (0.0, 5.5, 3.25, 0.0)
    ohne_zoll0 = (G.ART_P[1] * 5.5 + G.ART_P[2] * 3.25) / (G.ART_P[0] * (1 - G.P_ZOLL) + G.ART_P[1] + G.ART_P[2] + G.ART_P[3])
    phi_i = np.zeros((len(Xe), 5))
    phi_i[:, 4] = Xe[:, 4] - mu0
    V = np.zeros((len(Xe), 32))
    for i, x in enumerate(Xe):
        for k in range(32):
            if (k >> 4) & 1:
                V[i, k] = x[4]
            elif (k >> 0) & 1:
                V[i, k] = mittel_art[int(x[0])]
            elif (k >> 2) & 1:
                V[i, k] = 0.0 if x[2] == 1 else ohne_zoll0
            else:
                V[i, k] = mu0
    return phi_i, shapley_np(V, 5)
