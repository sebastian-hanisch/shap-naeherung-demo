"""Orakel KernelSHAP und Möbius (ORACLE-PLAYBOOK §5): anderer Rechenweg als die Demo.

* KernelSHAP mit allen Koalitionen und Shapley-Gewicht gleich Aufzählung (LL17 Satz 2), an den eingefrorenen Spieltabellen **und** in Brüchen (siehe `test_handbeispiele.py`).
* Möbius-Transformation gegen Einschluss-Ausschluss; Shapley-Wert gleich Σ c_T/|T| (zweite Definition desselben Werts); Zeta-Transformation als Umkehrung.
* Gepaartes Ziehen exakt für Möbius-Ordnung ≤ 2 (jede nicht singuläre Ziehung), nicht für Ordnung 3: Eigenschaft über Zufallsspiele mit festem Zufalls-Seed.
* Shapley-Werte: schnelle gegen allgemeine Fassung gegen alle Permutationen an Zufallsspielen."""

import itertools
import math

import numpy as np
import pytest

import shn_data as DATA
import shn_kernel as K
import shn_moebius as MB
from shn_rng import SplitMix64
from shn_shapley import shapley_aus_werten, shapley_np


def _perm(v, d):
    phi = np.zeros(d)
    perms = list(itertools.permutations(range(d)))
    for p in perms:
        k, prev = 0, v[0]
        for j in p:
            k |= 1 << j
            phi[j] += v[k] - prev
            prev = v[k]
    return phi / len(perms)


def test_shapley_drei_fassungen_gleich_an_zufallsspielen():
    for seed in range(12):
        d = 5
        v = MB.zufallsspiel(d, d, SplitMix64(seed))
        assert np.allclose(shapley_np(v, d), [float(x) for x in shapley_aus_werten(list(v), d)], atol=1e-12)
        assert np.allclose(shapley_np(v, d), _perm(v, d), atol=1e-12)
        assert shapley_np(v, d).sum() == pytest.approx(v[-1] - v[0], abs=1e-12)           # Effizienz


@pytest.mark.parametrize("d", [6, 10])
def test_kernel_mit_allen_koalitionen_und_shapley_gewicht_gleich_aufzaehlung_an_den_tabellen(d):
    for eintrag in DATA.games(d)["modelle"]:
        V = np.array(eintrag["v"])
        ex = shapley_np(V[0, :2], d)
        for i in range(2):
            assert np.abs(K.kernel_voll_gewichtet(V[0, i], d) - ex[i]).max() < 1e-9, eintrag["modell"]


def test_moebius_gegen_einschluss_ausschluss_und_zeta_ist_die_umkehrung():
    d = 5
    for seed in range(8):
        v = MB.zufallsspiel(d, 3, SplitMix64(seed)) + 0.7                                # konstanter Term
        c = MB.moebius(v, d)
        for k in range(1 << d):
            T = [j for j in range(d) if (k >> j) & 1]
            direkt = sum((-1) ** (len(T) - len(U)) * v[sum(1 << j for j in U)] for r in range(len(T) + 1) for U in itertools.combinations(T, r))
            assert c[k] == pytest.approx(direkt, abs=1e-12)
        assert MB.zeta(c, d) == pytest.approx(v, abs=1e-12)
        assert MB.shapley_aus_moebius(c, d) == pytest.approx(shapley_np(v, d), abs=1e-12)    # φ_i = Σ_{T ∋ i} c_T / |T|
        assert MB.ordnung(c, d, 1e-9) == 3


def test_ordnung_und_anteile_an_handspielen():
    d = 4
    c = np.zeros(1 << d)
    c[0b0111] = 1.0                                                                       # nur c_{0,1,2} = 1: Ordnung 3
    v = MB.zeta(c, d)
    assert MB.ordnung(MB.moebius(v, d), d) == 3 and MB.anteile_je_ordnung(MB.moebius(v, d), d).tolist() == [0, 0, 1, 0]
    assert shapley_np(v, d) == pytest.approx([1 / 3, 1 / 3, 1 / 3, 0], abs=1e-12)
    additiv = MB.zeta(np.array([0, 1.0, 2.0, 0, 3.0, 0, 0, 0] + [0] * 8), d)
    assert MB.ordnung(MB.moebius(additiv, d), d) == 1


def test_gepaartes_ziehen_ist_exakt_fuer_ordnung_hoechstens_2_wenn_nicht_singulaer():
    d, m = 6, 12
    stat = {1: [0, 0], 2: [0, 0], 3: [0, 0]}                                              # [nicht singulär, davon exakt]
    for ordn in (1, 2, 3):
        for t in range(40):
            v = MB.zufallsspiel(d, ordn, SplitMix64(1000 * ordn + t))
            ex = shapley_np(v, d)
            mk = K.masken_ziehen(d, 2 * m, SplitMix64(77 + t), True)
            phi, sing = K.kernel_loesen(mk, v[K.masken_index(mk)], v[0], v[-1])
            if not sing:
                stat[ordn][0] += 1
                stat[ordn][1] += bool(np.abs(phi - ex).max() < 1e-9)
    assert stat[1][0] > 30 and stat[1][1] == stat[1][0]
    assert stat[2][0] > 30 and stat[2][1] == stat[2][0]
    assert stat[3][0] > 30 and stat[3][1] == 0                                            # Ordnung 3: in keinem nicht singulären Spiel exakt (Gegenprobe zur Absolutaussage)


def test_ungerader_teil_ist_halbe_phi_mal_s_bis_ordnung_2():
    d = 5
    for ordn, soll in ((1, True), (2, True), (3, False)):
        v = MB.zufallsspiel(d, ordn, SplitMix64(31 + ordn))
        phi = shapley_np(v, d)
        gleich = True
        for k in range(1 << d):
            o = (v[k] - v[(1 << d) - 1 - k]) / 2
            s = np.array([1 if (k >> j) & 1 else -1 for j in range(d)])
            gleich &= bool(abs(o - 0.5 * phi @ s) < 1e-12)
        assert gleich == soll
    # Handwert der Herleitung: z_i z_j = (s_i s_j + s_i + s_j + 1) / 4 für alle z in {0, 1}
    for zi, zj in itertools.product((0, 1), repeat=2):
        si, sj = 2 * zi - 1, 2 * zj - 1
        assert zi * zj == (si * sj + si + sj + 1) / 4


def test_alle_inneren_koalitionen_zaehlen_wie_in_der_geschlossenen_form():
    assert [2 ** d - 2 for d in (6, 10)] == [62, 1022]
    d = 6
    p = np.array(K.groessen_wahrscheinlichkeit(d))
    gew = [(d - 1) / (math.comb(d, s) * s * (d - s)) for s in range(1, d)]
    assert p == pytest.approx(np.array(gew) * np.array([math.comb(d, s) for s in range(1, d)]) / sum(g * math.comb(d, s) for s, g in zip(range(1, d), gew)), abs=1e-12)
