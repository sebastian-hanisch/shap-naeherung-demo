"""Kopie-Test: `shn_generator.py` ist eine Kopie der Teile von `verweilzeit-demo` (`vwz_generator.py`, `vwz_constants.py`), kein Import. Die Werte unten stammen aus der Quelle
(eingefroren am 2026-11-19 aus `vwz_generator`): sie fallen auf, sobald die Kopie vom Original abweicht. Der Generator ist erfunden (kein Echtdatenbezug)."""

import numpy as np
import pytest

import shn_generator as G


def test_kundeneffekte_wie_in_der_quelle():
    assert G.UK[:4].tolist() == pytest.approx([-0.42714751093638936, 0.3791185374387331, -0.2611985213877257, -0.07775197048031927], abs=1e-12)
    assert len(G.UK) == G.K_KUNDEN == 12


def test_bekannte_funktion_mu_wie_in_der_quelle():
    mu, s = G.log_median_and_s(np.array([0, 1, 2, 3]), np.array([0, 5, 7, 11]), np.array([1, 0, 0, 0]), np.array([1, 0, 1, 0]), np.array([0.0, 4.0, 2.5, 0.0]))
    assert mu.tolist() == pytest.approx([1.8256154575589787, 1.3196147424321858, 0.8913094751324775, 2.2364592207808904], abs=1e-12)
    assert s.tolist() == [0.55, 0.25, 0.35, 0.9]


def test_historie_wie_in_der_quelle():
    X, d = G.make_history(6, 3)
    assert X[:, :6] == pytest.approx(np.array([[1.0, 9.0, 0.0, 0.0, 4.529070782436092, -1.7397705836305408], [1.0, 8.0, 0.0, 0.0, 3.1242242502696564, 0.17573397879188135],
                                              [1.0, 7.0, 0.0, 0.0, 8.262606928312948, -0.5447547152222929], [1.0, 7.0, 0.0, 0.0, 2.183649719564464, 1.0092209494056046],
                                              [2.0, 11.0, 0.0, 0.0, 4.374534388532064, -0.9367790901195847], [1.0, 6.0, 0.0, 1.0, 3.202168777956296, 0.1752786654277327]]), abs=1e-12)
    assert d.tolist() == pytest.approx([4.150002402947675, 4.066343042284527, 7.023255016474208, 2.9663735656293757, 4.04571203381064, 2.4654792908344962], abs=1e-12)


def test_struktur_zoll_nur_bei_import_vorlauf_nur_bei_export_und_umschlag():
    X, _ = G.make_history(4000, 1)
    art, zoll, vor = X[:, 0], X[:, 2], X[:, 4]
    assert (zoll[art != 0] == 0).all() and zoll[art == 0].any()
    assert (vor[(art == 0) | (art == 3)] == 0).all()
    assert ((vor[art == 1] >= 1) & (vor[art == 1] <= 10)).all() and ((vor[art == 2] >= 0.5) & (vor[art == 2] <= 6)).all()


def test_daten_und_wahrheit_haengen_zusammen():
    X, y = G.daten(50, 2)
    assert X.shape == (50, 6) and y.shape == (50,)
    mu = G.wahrheit(X)
    assert mu.shape == (50,) and np.isfinite(mu).all()
