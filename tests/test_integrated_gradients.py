"""Einordnungsabsatz Integrated Gradients (aus `handbeispiele_2b.py` Teil 7): Handwerte, Quadraturregel, Gradient ohne Torch (numpy). Die App hat dafür keinen Rechencode;
dieser Test belegt die Zahlen, die der Absatz in `shn_texts.IG_ABSATZ` und die README nennen. Quelle der Handwerte: Sundararajan u. a. 2017, S. 5 (gelesen)."""

from fractions import Fraction as Fr

import numpy as np

from helpers_exact import shapley_koal


def test_min_x1_x2_ig_gibt_dem_kritischen_merkmal_alles_baseline_shapley_teilt():
    f = lambda a, b: min(a, b)
    x = (1.0, 3.0)
    m = 2000
    al = (np.arange(m) + 0.5) / m                                                          # Mittelpunktsregel
    g1 = np.where(al * x[0] < al * x[1], 1.0, 0.0)
    g2 = np.where(al * x[0] < al * x[1], 0.0, 1.0)
    ig = (x[0] * g1.mean(), x[1] * g2.mean())
    assert np.allclose(ig, (1.0, 0.0)) and abs(sum(ig) - (f(*x) - f(0, 0))) < 1e-12        # IG (1, 0), Vollständigkeit
    v = {frozenset(): Fr(0), frozenset({0}): Fr(0), frozenset({1}): Fr(0), frozenset({0, 1}): Fr(1)}
    assert shapley_koal(2, lambda S: v[frozenset(S)]) == [Fr(1, 2), Fr(1, 2)]               # Baseline-Shapley (1/2, 1/2)


def test_x1_x2_beide_x1x2_halbe():
    al = (np.arange(2000) + 0.5) / 2000
    xs = (1.5, 2.0)
    ig = (xs[0] * (al * xs[1]).mean(), xs[1] * (al * xs[0]).mean())
    assert np.allclose(ig, (xs[0] * xs[1] / 2, xs[0] * xs[1] / 2), atol=1e-6)


def test_quadraturregel_rechte_endpunkte_gegen_mittelpunkte():
    xx = np.array([1.5, -0.7])
    fx = xx[0] * xx[1] + np.sin(xx[0])
    soll = {5: (0.357, 3.75e-3), 20: (0.0878, 2.34e-4), 50: (0.0350, 3.74e-5), 200: (0.00874, 2.34e-6)}
    for m, (e_rechts, e_mitte) in soll.items():
        for a, e in (((np.arange(1, m + 1)) / m, e_rechts), ((np.arange(m) + 0.5) / m, e_mitte)):
            d1 = a * xx[1] + np.cos(a * xx[0])
            d2 = a * xx[0]
            err = abs(xx[0] * d1.mean() + xx[1] * d2.mean() - fx)
            assert abs(err - e) / e < 0.02, (m, err, e)
    # rechte Endpunkte etwa 1/m (Verhältnis der Fehler bei m = 50 und m = 200 etwa 4), Mittelpunkte etwa 1/m² (etwa 16)
    assert 3.5 < 0.0350 / 0.00874 < 4.5 and 14 < 3.74e-5 / 2.34e-6 < 17


def test_gradient_eines_relu_netzes_in_numpy_gegen_zentrale_differenzen():
    rng = np.random.default_rng(3)
    W1, b1, W2, b2 = rng.normal(size=(3, 5)), rng.normal(size=5), rng.normal(size=(5, 1)), rng.normal(size=1)
    net = lambda X: (np.maximum(X @ W1 + b1, 0) @ W2 + b2).ravel()
    X0 = rng.normal(size=(4, 3))
    grad = (((X0 @ W1 + b1) > 0) * W2.ravel()) @ W1.T
    h = 1e-6
    fd = np.column_stack([(net(X0 + h * np.eye(3)[j]) - net(X0 - h * np.eye(3)[j])) / (2 * h) for j in range(3)])
    assert np.abs(grad - fd).max() < 1e-6
