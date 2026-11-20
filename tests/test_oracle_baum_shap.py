"""Orakel Baum-SHAP (ORACLE-PLAYBOOK §5): das pfadabhängige v(S) und der Shapley-Wert auf **mehreren unabhängigen Rechenwegen** an trainierten scikit-learn-Bäumen.

Rechenwege: (a) vektorisierte Blattrechnung (`blatt_rechnung`), (b) Knotenrekursion der Demo (Aufzählung, Algorithmus 1), (c) eigene Rekursion hier auf den Feldern von
scikit-learn (`weighted_n_node_samples`), (d) Algorithmus 2, (e) alle d! Permutationen (d = 6). Dazu v(alle) = Vorhersage und Effizienz."""

import itertools
import sys
from pathlib import Path

import numpy as np
import pytest

sk = pytest.importorskip("sklearn")
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor  # noqa: E402
from sklearn.tree import DecisionTreeRegressor  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools"))
import shn_generator as G  # noqa: E402
import shn_make_common as MC  # noqa: E402
from shn_shapley import shapley_aus_werten, shapley_np  # noqa: E402
from shn_tree import blatt_rechnung, werte_aufzaehlung  # noqa: E402
from shn_treeshap import treeshap_poly  # noqa: E402


def _rekursion_sklearn(tree, x, S):
    """v(S) direkt auf den Feldern des scikit-learn-Baums (Knotenrekursion, unabhängig von shn_tree)."""
    t = tree.tree_

    def rec(i):
        if t.children_left[i] == -1:
            return float(t.value[i][0][0])
        f, a, c = t.feature[i], t.children_left[i], t.children_right[i]
        if f in S:
            return rec(a) if x[f] <= t.threshold[i] else rec(c)
        return (t.weighted_n_node_samples[a] * rec(a) + t.weighted_n_node_samples[c] * rec(c)) / t.weighted_n_node_samples[i]
    return rec(0)


def _permutationen(vals, d):
    phi = np.zeros(d)
    perms = list(itertools.permutations(range(d)))
    for p in perms:
        k, prev = 0, vals[0]
        for j in p:
            k |= 1 << j
            phi[j] += vals[k] - prev
            prev = vals[k]
    return phi / len(perms)


@pytest.mark.parametrize("seed,tiefe", [(0, 2), (1, 4), (2, 6), (3, 3)])
def test_vier_rechenwege_geben_dasselbe_v_und_phi(seed, tiefe):
    X, y = G.daten(600, 300 + seed)
    t = DecisionTreeRegressor(max_depth=tiefe, min_samples_leaf=10, random_state=seed).fit(X, y)
    b = MC.baum_aus_sklearn(t)
    Xe = G.daten(4, 6000 + seed)[0]
    vb = blatt_rechnung(b, Xe, 6)
    for i, x in enumerate(Xe):
        vals, _ = werte_aufzaehlung(b, x, 6)
        assert np.allclose(vb[i], vals, atol=1e-12)                                      # (a) gegen (b)
        for k in range(0, 64, 7):
            S = {j for j in range(6) if (k >> j) & 1}
            assert _rekursion_sklearn(t, x, S) == pytest.approx(vals[k], abs=1e-12)      # (c) gegen (b)
        phi_koal = shapley_np(np.array(vals)[None, :], 6)[0]
        phi2, _ = treeshap_poly(b, x, 6)
        assert np.allclose(phi2, phi_koal, atol=1e-10)                                   # (d)
        assert np.allclose(_permutationen(np.array(vals), 6), phi_koal, atol=1e-10)      # (e)
        assert np.allclose(phi_koal, shapley_aus_werten(vals, 6), atol=1e-12)           # schnelle gegen allgemeine Fassung
        assert vals[-1] == pytest.approx(t.predict(x[None])[0], abs=1e-12)               # v(alle) = Vorhersage
        assert sum(phi2) == pytest.approx(t.predict(x[None])[0] - vals[0], abs=1e-10)    # Effizienz


def test_ensemble_v_alle_ist_vorhersage_und_blattanteile_stimmen():
    X, y = G.daten(500, 41)
    for m in (GradientBoostingRegressor(n_estimators=30, max_depth=3, random_state=0), RandomForestRegressor(15, min_samples_leaf=3, max_features=0.5, random_state=0)):
        m.fit(X[:, :5], y)
        V = MC.pfad_werte(m, X[:7, :5], 5)                                               # prüft v(alle) = Vorhersage selbst (assert)
        assert V.shape == (7, 32)
    t = DecisionTreeRegressor(max_depth=4, min_samples_leaf=5, random_state=0).fit(X, y)
    b = MC.baum_aus_sklearn(t)
    assert b.cover == t.tree_.weighted_n_node_samples.tolist()                          # Blattanteile gegen scikit-learn
    assert all(abs(b.cover[i] - (b.cover[b.left[i]] + b.cover[b.right[i]])) < 1e-9 for i in range(b.n_knoten) if b.left[i] != -1)


def test_algorithmus2_bei_wiederholten_merkmalen_im_pfad():
    """Tiefe 8 über zwei Merkmalen: jedes Merkmal kommt in jedem Pfad mehrfach vor."""
    rng = np.random.default_rng(5)
    X = rng.uniform(-1, 1, size=(800, 2))
    y = np.sin(5 * X[:, 0]) + np.cos(4 * X[:, 1]) + X[:, 0] * X[:, 1]
    t = DecisionTreeRegressor(max_depth=8, min_samples_leaf=3, random_state=0).fit(X, y)
    b = MC.baum_aus_sklearn(t)
    for x in X[:5]:
        vals, _ = werte_aufzaehlung(b, x, 2)
        phi2, _ = treeshap_poly(b, x, 2)
        assert np.allclose(phi2, shapley_np(np.array(vals)[None, :], 2)[0], atol=1e-10)
        assert np.allclose(phi2, _permutationen(np.array(vals), 2), atol=1e-10)
