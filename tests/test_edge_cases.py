"""Randfälle (DEMO-PLAYBOOK §12): Baum mit einem Blatt, d = 1 und 2, M = 2 und alle Koalitionen, weniger als d − 1 Paare (Meldung statt Zahl), Hintergrund mit einer Zeile,
nicht benutzte Merkmale, q am Rand, Zufalls-Seeds."""

from fractions import Fraction as Fr

import numpy as np
import pytest

import shn_evaluation as ev
import shn_experiments as E
import shn_kernel as K
import shn_moebius as MB
import shn_toy as T
from shn_rng import SplitMix64
from shn_shapley import shapley_aus_werten, shapley_np
from shn_tree import Baum, werte_aufzaehlung
from shn_treeshap import treeshap_poly


def test_baum_mit_einem_blatt_alle_beitraege_null():
    b = Baum.aus_tupel(("L", 2.5))
    assert b.tiefe() == 0 and b.n_blaetter() == 1
    vals, besuche = werte_aufzaehlung(b, (1, 0, 1), 3)
    assert vals == [2.5] * 8 and besuche == 8                                             # jede der 2^3 Koalitionen besucht das eine Blatt
    assert shapley_aus_werten(vals, 3) == [0, 0, 0]
    phi, schritte = treeshap_poly(b, (1, 0, 1), 3)
    assert phi == [0, 0, 0] and schritte == 1


def test_ein_merkmal_beitrag_ist_der_ganze_zuwachs():
    b = Baum.aus_tupel(("N", 0, 0, ("L", Fr(2)), ("L", Fr(6)), Fr(3), Fr(1)))             # Mittel 3, x = 1 gibt 6
    vals, besuche = werte_aufzaehlung(b, (1,), 1)
    assert vals == [Fr(3), Fr(6)] and besuche == 5
    assert shapley_aus_werten(vals, 1) == [Fr(3)] == treeshap_poly(b, (1,), 1, Fr(1))[0]


def test_nicht_benutzte_merkmale_bekommen_null_pfadabhaengig():
    b = Baum.aus_tupel(("N", 1, 0, ("L", Fr(0)), ("L", Fr(4)), Fr(1), Fr(1)))             # liest nur Merkmal 1 von dreien
    phi, _ = treeshap_poly(b, (0, 1, 1), 3, Fr(1))
    assert phi[0] == 0 and phi[2] == 0 and phi[1] == Fr(2)


def test_d_gleich_2_kernel_mit_den_beiden_einzelkoalitionen():
    v = np.array([0.0, 1.0, 3.0, 5.0])                                                    # v({0}) = 1, v({1}) = 3, v(alle) = 5
    mk = np.array([[1, 0], [0, 1]], dtype=bool)
    phi, sing = K.kernel_loesen(mk, v[K.masken_index(mk)], v[0], v[-1])
    assert not sing and phi.sum() == pytest.approx(5.0)
    assert K.kernel_voll_gewichtet(v, 2) == pytest.approx(shapley_np(v, 2), abs=1e-12)


def test_zwei_koalitionen_gepaart_ist_singulaer_und_das_verdikt_meldet_es():
    r = E.kernel_experiment(6, "Boosting Tiefe 3", 2, True, 3)
    assert r["singulaer"] == r["n_ziehungen"] == 15                                      # ein Paar kann d − 1 = 5 Unbekannte nie festlegen
    kind, text = ev.verdict_kernel(r)
    assert "singulär" in text and kind in ("info", "warning")


def test_weniger_als_d_minus_1_paare_melden_singulaer_statt_zahl():
    r = E.zufallsspiele(6, 2, 4)                                                          # 4 < d − 1 = 5
    assert r["singulaer"] == r["ziehungen"] == 200 and max(r["anteile"]) == 0
    kind, text = ev.verdict_zufallsspiele(r)
    assert kind == "warning" and "singulär" in text and "Meldung statt einer Zahl" in text


def test_ein_paar_bei_d_2_reicht_fuer_die_eine_unbekannte():
    v = MB.zufallsspiel(2, 2, SplitMix64(1))
    mk = K.masken_ziehen(2, 2, SplitMix64(1), True)
    phi, sing = K.kernel_loesen(mk, v[K.masken_index(mk)], v[0], v[-1])
    assert not sing and phi == pytest.approx(shapley_np(v, 2), abs=1e-12)


def test_hintergrund_mit_einer_zeile_ist_ein_gewoehnliches_spiel():
    f = lambda z: 2 * z[0] + z[1] * z[2]
    x, b = np.array([1.0, 2.0, 3.0]), np.array([0.0, 1.0, 1.0])
    v = np.array([f(np.where([(k >> j) & 1 for j in range(3)], x, b)) for k in range(8)])
    phi = shapley_np(v, 3)
    assert phi.sum() == pytest.approx(f(x) - f(b), abs=1e-12) and phi[0] == pytest.approx(2.0)   # 2·x1 ist additiv: Beitrag 2·(1 − 0)


def test_q_am_rand_und_unabhaengig():
    assert E.exp1(0.5, 0, "pfadabhängig")["klassen"][0][1] == 131 and len(E.exp1(0.5, 0, "pfadabhängig")["klassen"]) == 1
    r = E.exp1(0.99, 5, "pfadabhängig")
    assert r["n_formen"] == 131 and sum(c for _, c in r["klassen"]) == 131


def test_funktional_interventional_und_konditional_sind_fuer_alle_formen_gleich():
    for fkt, erwartet in (("interventional", [0.5, 0.0, 0.0]), ("konditional", [0.34, 0.2, 0.0])):
        r = E.exp1(0.9, 17, fkt)
        assert len(r["klassen"]) == 1 and r["klassen"][0][1] == 131 and r["wichtigkeit"] == pytest.approx(erwartet, abs=1e-9)
        assert r["modell_gleich_f"]


def test_seeds_aus_mehreren_zahlen_sind_fest_und_verschieden():
    assert E.seed_aus(6, 0, 1, 24, 1) == E.seed_aus(6, 0, 1, 24, 1)
    assert len({E.seed_aus(6, s, z, 24, 1) for s in range(5) for z in range(10)}) == 50


def test_moebius_des_leeren_und_konstanten_spiels():
    assert MB.ordnung(MB.moebius(np.full(16, 3.0), 4), 4) == 0                              # konstantes Spiel hat keine Ordnung
    assert MB.anteile_je_ordnung(MB.moebius(np.full(16, 3.0), 4), 4).tolist() == [0, 0, 0, 0]


def test_form_ohne_wurzel_und_masse_bei_einem_blatt():
    b = Baum.aus_tupel(("L", Fr(1)))
    assert T.wurzel_merkmal(b) is None and T.masse_x2_vor_x1(b) == 0
