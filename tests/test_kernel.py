"""KernelSHAP-Bausteine: Ziehen (Größenverteilung, Paare), Regression (Nebenbedingung, Singularität), SplitMix64-Zufall."""

import numpy as np
import pytest

import shn_kernel as K
from shn_rng import SplitMix64


def test_gezogene_groessen_liegen_zwischen_1_und_d_minus_1():
    rng = SplitMix64(3)
    for d in (2, 3, 6, 10):
        mk = K.masken_ziehen(d, 200, rng, False)
        assert mk.shape == (200, d) and mk.sum(axis=1).min() >= 1 and mk.sum(axis=1).max() <= d - 1


def test_gepaart_ist_koalition_und_komplement():
    mk = K.masken_ziehen(6, 24, SplitMix64(5), True)
    assert mk.shape == (24, 6) and (mk[:12] == ~mk[12:]).all()


def test_groessenverteilung_folgt_dem_shapley_kern_im_band():
    d, n = 6, 20000
    mk = K.masken_ziehen(d, n, SplitMix64(11), False)
    haeufig = np.bincount(mk.sum(axis=1), minlength=d + 1)[1:d] / n
    p = np.array(K.groessen_wahrscheinlichkeit(d))
    assert p.sum() == pytest.approx(1.0)
    se = np.sqrt(p * (1 - p) / n)
    assert (np.abs(haeufig - p) < 4 * se).all()                                          # Band mit Standardfehler gegen die exakte Größe
    assert p[0] == p[-1] and p[0] > p[1]                                                 # symmetrisch, kleine und große Koalitionen am häufigsten


def test_dieselben_seeds_geben_dieselben_ziehungen():
    a = K.masken_ziehen(10, 40, SplitMix64(9), True)
    b = K.masken_ziehen(10, 40, SplitMix64(9), True)
    c = K.masken_ziehen(10, 40, SplitMix64(10), True)
    assert (a == b).all() and not (a == c).all()


def test_fehler_bei_zu_kleinem_d_und_ungeradem_m_gepaart():
    with pytest.raises(ValueError):
        K.masken_ziehen(1, 4, SplitMix64(1), False)
    with pytest.raises(ValueError):
        K.masken_ziehen(4, 5, SplitMix64(1), True)


def test_regression_erfuellt_die_nebenbedingung_und_gibt_additives_spiel_exakt():
    d = 5
    c = np.array([0.5, -1.0, 2.0, 0.25, 3.0])
    v = np.array([sum(c[j] for j in range(d) if (k >> j) & 1) for k in range(1 << d)])      # additives Spiel: φ = c
    mk = K.masken_ziehen(d, 20, SplitMix64(2), True)
    phi, sing = K.kernel_loesen(mk, v[K.masken_index(mk)], v[0], v[-1])
    assert not sing and phi == pytest.approx(c, abs=1e-10) and phi.sum() == pytest.approx(v[-1] - v[0], abs=1e-12)


def test_ein_paar_ist_singulaer_und_wird_gemeldet():
    d = 6
    mk = np.array([[1, 0, 0, 0, 0, 0], [0, 1, 1, 1, 1, 1]], dtype=bool)                # ein Paar: Koalition und Komplement
    v = np.arange(1 << d, dtype=float)
    _, sing = K.kernel_loesen(mk, v[K.masken_index(mk)], v[0], v[-1])
    assert sing


def test_kosten_in_modellauswertungen():
    assert K.kosten_auswertungen(24, 30) == 720 and K.kosten_auswertungen(40, 30) == 1200


def test_splitmix_ist_bitgleich_fest_und_gleichverteilt_im_band():
    assert SplitMix64(0).next_u64() == 0xE220A8397B1DCDAF                                # bekannter erster Ausgabewert von SplitMix64 mit Seed 0
    rng = SplitMix64(4)
    xs = np.array([rng.random() for _ in range(20000)])
    assert abs(xs.mean() - 0.5) < 4 * np.sqrt(1 / 12 / 20000)
