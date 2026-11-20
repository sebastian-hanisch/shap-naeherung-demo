"""Handbeispiele von Hand, exakt in Brüchen (aus `handbeispiele_2b.py` der Planung übernommen, hier gegen die Module der Demo; die Orakel in `helpers_exact.py` rechnen unabhängig).

Teile: 1 Mini-Baum (zwei Merkmale), 2 alle 131 Baumformen, 3 Algorithmus 2 gegen Aufzählung, 4 Kosten in Zählungen, 5 KernelSHAP in Brüchen, 6 Möbius-Ordnung von Baumspielen.
Alles hier ist erfunden (kleine Tabellen, Brüche); Literatur nur für Stufe W (TS20 Algorithmus 1 und 2, LL17 Satz 2). Die Herleitung in Teil 5 ist eigene Rechnung."""

import itertools
import random
from fractions import Fraction as Fr
from math import comb

import pytest

import helpers_exact as H
import shn_experiments as E
import shn_moebius as MB
import shn_toy as T
from shn_shapley import shapley_aus_werten
from shn_tree import Baum, werte_aufzaehlung
from shn_treeshap import besuche_vollstaendiger_baum, schritte_vollstaendiger_baum, treeshap_poly


def _phi_aufz(b, x, d):
    return shapley_aus_werten(werte_aufzaehlung(b, x, d)[0], d)


# ---------------------------------------------------------------------------------------------- Teil 1: Mini-Baum von Hand
def test_teil1_mini_baum_von_hand():
    pr = {(0, 0): Fr(9, 20), (1, 1): Fr(9, 20), (0, 1): Fr(1, 20), (1, 0): Fr(1, 20)}
    L0, L1 = ("L", Fr(0)), ("L", Fr(1))
    A = ("N", 0, 0, L0, L1, Fr(1, 2), Fr(1, 2))
    B = ("N", 1, 0, ("N", 0, 0, L0, L1, pr[(0, 0)], pr[(1, 0)]), ("N", 0, 0, L0, L1, pr[(0, 1)], pr[(1, 1)]), Fr(1, 2), Fr(1, 2))
    x = (1, 1)
    for t in (A, B):
        assert all(H.alg1_tupel(t, p, frozenset({0, 1})) == p[0] for p in pr)           # beide berechnen f = x1
    # Von Hand: Baum A: v(leer) = 1/2, v({1}) = 1, v({2}) = 1/2 (x2 wird nicht gelesen), v({1,2}) = 1; φ1 = ½·½ + ½·½ = 1/2, φ2 = 0
    va = {frozenset(): Fr(1, 2), frozenset({0}): Fr(1), frozenset({1}): Fr(1, 2), frozenset({0, 1}): Fr(1)}
    for S, w in va.items():
        assert H.alg1_tupel(A, x, S) == w
    # Baum B: v({2}) = 9/10; φ1 = ½·½ + ½·(1/10) = 3/10, φ2 = ½·(2/5) + 0 = 1/5
    vb = {frozenset(): Fr(1, 2), frozenset({0}): Fr(1), frozenset({1}): Fr(9, 10), frozenset({0, 1}): Fr(1)}
    for S, w in vb.items():
        assert H.alg1_tupel(B, x, S) == w
    mini = E.mini_beispiel()                                                          # Code der Demo
    assert mini["A"]["phi"] == [Fr(1, 2), 0] and mini["B"]["phi"] == [Fr(3, 10), Fr(1, 5)]
    assert H.shapley_perm(2, lambda S: H.alg1_tupel(B, x, S)) == [Fr(3, 10), Fr(1, 5)]
    assert mini["A"]["wichtigkeit"] == [Fr(1, 2), 0] and mini["B"]["wichtigkeit"] == [Fr(17, 50), Fr(1, 5)]
    # Fremdreferenz: Wahrheit der Funktion aus der Verteilung (kein Baum): konditional = Baum B, interventional = Baum A
    def f(z):
        return Fr(z[0])
    vk = lambda S: (sum(p * f(b) for b, p in pr.items() if all(b[j] == x[j] for j in S)) / sum(p for b, p in pr.items() if all(b[j] == x[j] for j in S)))
    vi = lambda S: sum(p * f(tuple(x[j] if j in S else b[j] for j in range(2))) for b, p in pr.items())
    assert H.shapley_koal(2, vk) == [Fr(3, 10), Fr(1, 5)] and H.shapley_koal(2, vi) == [Fr(1, 2), 0]


# ---------------------------------------------------------------------------------------------- Teil 2: alle 131 Baumformen
def test_teil2_131_baumformen_und_klassen():
    eigene = H.alle_baeume()                                                          # Aufzählung auf Tupeln, ohne die Demo
    demo = T.baumformen(Fr(9, 10))
    assert len(eigene) == len(demo) == 131
    nach_tiefe = {}
    for t in eigene:
        nach_tiefe[H.tiefe(t)] = nach_tiefe.get(H.tiefe(t), 0) + 1
    assert nach_tiefe == {1: 1, 2: 10, 3: 120}
    assert sorted(b.tiefe() for b in demo) == sorted(H.tiefe(t) for t in eigene)
    tripel, regel_ok = {}, 0
    for i, t in enumerate(eigene):
        b = Baum.aus_tupel(t)
        ph = {x: shapley_aus_werten(werte_aufzaehlung(b, x, 3)[0], 3) for x in H.PTS}
        for x in H.PTS:
            assert sum(ph[x]) == x[0] - H.alg1_tupel(t, x, frozenset())                 # Effizienz
            assert H.alg1_tupel(t, x, frozenset(range(3))) == x[0]                      # v(alle) = f(x)
        if i % 7 == 0:
            for x in H.PTS:
                assert ph[x] == H.shapley_perm(3, lambda S, x=x: H.alg1_tupel(t, x, S))  # Koalitionsformel = alle 6 Permutationen
        mean = tuple(sum(H.PR[x] * abs(ph[x][j]) for x in H.PTS) for j in range(3))
        tripel.setdefault(mean, []).append(t)
        m_vor = sum(m for pf, m in H.blatt_pfade(t) if 1 in pf and pf.index(1) < pf.index(0))
        regel_ok += mean[1] == m_vor / 5
    assert {k: len(v) for k, v in tripel.items()} == {(Fr(1, 2), 0, 0): 97, (Fr(17, 50), Fr(1, 5), 0): 26, (Fr(21, 50), Fr(1, 10), Fr(1, 30)): 8}
    assert regel_ok == 131                                                              # Beitrag x2 = Masse(x2 vor x1) / 5
    # derselbe Befund mit dem Code der Demo (Brüche)
    klassen = T.klassen_zaehlen([T.wichtigkeit(T.beitraege_pfad(b, Fr(9, 10)), Fr(9, 10)) for b in demo])
    assert [(round(w[1], 6), c) for w, c in klassen] == [(0.0, 97), (0.1, 8), (0.2, 26)]
    assert sum(1 for b in demo if T.masse_x2_vor_x1(b) / 5 == T.wichtigkeit(T.beitraege_pfad(b, Fr(9, 10)), Fr(9, 10))[1]) == 131
    wurzel = {}
    for k, ts in tripel.items():
        for t in ts:
            key = (k[1], t[1] if t[0] == "N" else None)
            wurzel[key] = wurzel.get(key, 0) + 1
    assert wurzel == {(Fr(0), 0): 81, (Fr(0), 2): 16, (Fr(1, 5), 1): 25, (Fr(1, 5), 2): 1, (Fr(1, 10), 2): 8}
    # Fremdreferenz: Wahrheit der Funktion aus der Verteilung, kein Baum
    f = lambda p: Fr(p[0])
    vi = lambda x, S: sum(H.PR[b] * f(tuple(x[j] if j in S else b[j] for j in range(3))) for b in H.PTS)
    vk = lambda x, S: sum(H.PR[b] * f(b) for b in H.PTS if all(b[j] == x[j] for j in S)) / sum(H.PR[b] for b in H.PTS if all(b[j] == x[j] for j in S))
    mi = [sum(H.PR[x] * abs(H.shapley_koal(3, lambda S, x=x: vi(x, S))[j]) for x in H.PTS) for j in range(3)]
    mk = [sum(H.PR[x] * abs(H.shapley_koal(3, lambda S, x=x: vk(x, S))[j]) for x in H.PTS) for j in range(3)]
    assert mi == [Fr(1, 2), 0, 0] and mk == [Fr(17, 50), Fr(1, 5), 0]
    assert T.wichtigkeit(T.wahrheit_interventional(Fr(9, 10)), Fr(9, 10)) == mi and T.wichtigkeit(T.wahrheit_konditional(Fr(9, 10)), Fr(9, 10)) == mk


def test_teil2_klassen_gelten_auch_fuer_andere_q():
    """Die Gegenprobe zu „97 / 26 / 8“: die Zählung hängt nicht an q = 9/10, solange x2 von x1 abhängt (Gleitkomma, vier Werte von q)."""
    for q in (0.6, 0.75, 0.9, 0.99):
        klassen = T.klassen_zaehlen(E.wichtigkeiten_pfad(q))
        assert sorted(c for _, c in klassen) == [8, 26, 97], q
    assert len(T.klassen_zaehlen(E.wichtigkeiten_pfad(0.5))) == 1                          # q = 1/2: x2 unabhängig, alle Formen gleich


# ---------------------------------------------------------------------------------------------- Teil 3: Algorithmus 2 gegen Aufzählung
def test_teil3_algorithmus2_gleich_aufzaehlung_in_brueche():
    n = 0
    for t in H.alle_baeume():
        b = Baum.aus_tupel(t)
        for x in H.PTS:
            ref = H.shapley_koal(3, lambda S, x=x: H.alg1_tupel(t, x, S))              # Aufzählung auf Tupeln (unabhängig)
            assert treeshap_poly(b, x, 3, Fr(1))[0] == ref
            n += 1
    assert n == 131 * 8


def test_teil3_wiederholtes_merkmal_im_pfad_und_leiter():
    w = {(0, 0): Fr(1, 10), (0, 1): Fr(2, 10), (1, 0): Fr(1, 10), (1, 1): Fr(2, 10), (2, 0): Fr(3, 10), (2, 1): Fr(1, 10)}
    assert sum(w.values()) == 1
    tl = ("N", 0, 1, ("L", Fr(2)), ("L", Fr(5)), w[(1, 0)], w[(2, 0)])
    tr = ("N", 1, 0, tl, ("L", Fr(7)), w[(1, 0)] + w[(2, 0)], w[(1, 1)] + w[(2, 1)])
    t = ("N", 0, 0, ("L", Fr(1)), tr, w[(0, 0)] + w[(0, 1)], w[(1, 0)] + w[(2, 0)] + w[(1, 1)] + w[(2, 1)])
    assert any(pf.count(0) == 2 for pf, _ in H.blatt_pfade(t))                          # Merkmal 0 zweimal im Pfad
    b = Baum.aus_tupel(t)
    for x in w:
        ref = H.shapley_koal(2, lambda S, x=x: H.alg1_tupel(t, x, S))
        assert treeshap_poly(b, x, 2, Fr(1))[0] == ref == H.shapley_perm(2, lambda S, x=x: H.alg1_tupel(t, x, S))
        assert sum(ref) == H.alg1_tupel(t, x, frozenset({0, 1})) - H.alg1_tupel(t, x, frozenset())
    assert H.alg1_tupel(t, (2, 0), frozenset({0, 1})) == 5                              # Handwert: Pfad Wurzel -> x1 = 0 -> Schwelle 1 -> Blatt 5
    m0 = {k: Fr(1, 4) for k in range(4)}
    leiter = ("N", 0, 2, ("N", 0, 1, ("N", 0, 0, ("L", Fr(0)), ("L", Fr(1)), m0[0], m0[1]), ("L", Fr(3)), m0[0] + m0[1], m0[2]), ("L", Fr(10)), m0[0] + m0[1] + m0[2], m0[3])
    bl = Baum.aus_tupel(leiter)
    for x0 in range(4):
        assert treeshap_poly(bl, (x0,), 1, Fr(1))[0] == H.shapley_koal(1, lambda S, x0=x0: H.alg1_tupel(leiter, (x0,), S))
    assert treeshap_poly(bl, (3,), 1, Fr(1))[0] == [Fr(10) - Fr(1, 4) * (0 + 1 + 3 + 10)]    # Handwert: φ = f(x) − Mittel = 10 − 7/2


# ---------------------------------------------------------------------------------------------- Teil 4: Kosten in Zählungen
def test_teil4_zaehlregel_von_hand_und_geschlossene_formen():
    # Von Hand: Baum mit einem Merkmal und zwei Blättern (d = 1): Aufzählung 3 + 2 = 5; Algorithmus 2: Wurzel 1 Aufruf + 2 Blätter je (1 + 1 + 1) = 7.
    # d = 2 (vollständig): Aufzählung 3^3 − 2^3 = 19; Algorithmus 2: 1 + 4 + 12 + 16 = 33.
    for d, (s1, s2) in {1: (5, 7), 2: (19, 33)}.items():
        b = Baum.aus_tupel(H.vollstaendiger_baum(d))
        x = (1,) * d
        _, visits = werte_aufzaehlung(b, x, d)
        ph, steps = treeshap_poly(b, x, d, Fr(1))
        assert (visits, steps) == (s1, s2)
        assert ph == shapley_aus_werten(werte_aufzaehlung(b, x, d)[0], d)
    for d in range(1, 7):
        b = Baum.aus_tupel(H.vollstaendiger_baum(d))
        x = (1,) * d
        assert werte_aufzaehlung(b, x, d)[1] == 3 ** (d + 1) - 2 ** (d + 1) == besuche_vollstaendiger_baum(d)
        assert treeshap_poly(b, x, d, Fr(1))[1] == sum(2 ** k * (1 + k) for k in range(d + 1)) + 2 ** d * d * d == schritte_vollstaendiger_baum(d)
    assert [schritte_vollstaendiger_baum(d) for d in (1, 2, 3, 8)] == [7, 33, 121, 20481]
    assert [besuche_vollstaendiger_baum(d) for d in (1, 2, 3, 8)] == [5, 19, 65, 19171]
    assert besuche_vollstaendiger_baum(10) == 175099 and schritte_vollstaendiger_baum(10) == 122881
    # KernelSHAP-Kosten in Modellauswertungen (Koalitionen mal 30 Hintergrundzeilen)
    assert Fr(40 * 30, (1 << 10) * 30) == Fr(5, 128) and Fr(24 * 30, (1 << 6) * 30) == Fr(3, 8)
    assert (40 * 30, (1 << 10) * 30, 24 * 30, (1 << 6) * 30) == (1200, 30720, 720, 1920) and [2 ** d - 2 for d in (6, 10)] == [62, 1022]


# ---------------------------------------------------------------------------------------------- Teil 5: KernelSHAP in Brüchen
def test_teil5_kernel_alle_koalitionen_mit_shapley_gewicht_ist_exakt():
    import shn_kernel as K
    import numpy as np
    d = 4
    rnd = random.Random(20261119)
    inner = [frozenset(c) for r in range(1, d) for c in itertools.combinations(range(d), r)]
    wk = [Fr(d - 1, comb(d, len(S)) * len(S) * (d - len(S))) for S in inner]
    assert len(inner) == 14
    for order in (1, 2, 3):
        for _ in range(15):
            coef = {T_: Fr(rnd.randint(-5, 5)) for r in range(1, order + 1) for T_ in itertools.combinations(range(d), r)}
            v = H.spiel(d, coef)
            ex = H.shapley_koal(d, lambda S: v[frozenset(S)])
            assert H.kernel_bruch(v, d, inner, wk) == ex                                 # LL17 Satz 2, in Brüchen
            assert ex == [sum((c / len(T_) for T_, c in coef.items() if i in T_), Fr(0)) for i in range(d)]   # zweite Definition: Σ c_T / |T|
            vec = np.array([float(v[frozenset(i for i in range(d) if (k >> i) & 1)]) for k in range(1 << d)])
            assert np.allclose(K.kernel_voll_gewichtet(vec, d), [float(e) for e in ex], atol=1e-9)    # Code der Demo (Gleitkomma)


def test_teil5_ungerader_teil_und_gepaartes_ziehen():
    d = 4
    rnd = random.Random(20261119)
    for order, soll in ((1, True), (2, True), (3, False)):
        alle_gleich = True
        for _ in range(15):
            coef = {T_: Fr(rnd.randint(-5, 5)) for r in range(1, order + 1) for T_ in itertools.combinations(range(d), r)}
            v = H.spiel(d, coef)
            ex = H.shapley_koal(d, lambda S: v[frozenset(S)])
            for S in v:
                comp = frozenset(range(d)) - S
                o = (v[S] - v[comp]) / 2
                lin = sum(ex[i] * (1 if i in S else -1) for i in range(d)) / 2
                alle_gleich &= o == lin
        assert alle_gleich == soll, f"Ordnung {order}"                                  # ungerader Teil = (1/2) φ·s genau bis Ordnung 2
    alle_paare = [frozenset(c) for r in (1, 2) for c in itertools.combinations(range(d), r)]
    stats = {1: [0, 0, 0], 2: [0, 0, 0], 3: [0, 0, 0]}                                  # [lösbar, exakt, singulär]
    for order in (1, 2, 3):
        for _ in range(60):
            coef = {T_: Fr(rnd.randint(1, 5)) * rnd.choice((-1, 1)) for r in range(1, order + 1) for T_ in itertools.combinations(range(d), r)}
            v = H.spiel(d, coef)
            ex = H.shapley_koal(d, lambda S: v[frozenset(S)])
            wahl = [rnd.choice(alle_paare) for _ in range(rnd.randint(1, 6))]
            design = wahl + [frozenset(range(d)) - S for S in wahl]
            try:
                ph = H.kernel_bruch(v, d, design, [1] * len(design))
            except ZeroDivisionError:
                stats[order][2] += 1
                continue
            stats[order][0] += 1
            stats[order][1] += ph == ex
    assert stats[1][0] > 0 and stats[1][0] == stats[1][1] and stats[2][0] > 0 and stats[2][0] == stats[2][1]   # Ordnung 1 und 2: exakt in allen lösbaren Ziehungen
    assert stats[3][0] > 0 and stats[3][1] < stats[3][0]                                # Ordnung 3: nicht in allen
    assert stats[2][2] > 0                                                              # zu wenige oder gleiche Paare: singulär
    for m in (1, 2):                                                                    # mit 1 oder 2 verschiedenen Paaren immer singulär (d = 4: drei Unbekannte)
        for wahl in itertools.combinations(alle_paare, m):
            design = list(wahl) + [frozenset(range(d)) - S for S in wahl]
            with pytest.raises(ZeroDivisionError):
                H.kernel_bruch(H.spiel(d, {(0,): Fr(1)}), d, design, [1] * len(design))


def test_teil5_handwerte_ordnung_3_und_2():
    d = 4
    inner = [frozenset(c) for r in range(1, d) for c in itertools.combinations(range(d), r)]
    v = H.spiel(d, {(0, 1, 2): Fr(1)})
    ex = H.shapley_koal(d, lambda S: v[frozenset(S)])
    assert ex == [Fr(1, 3), Fr(1, 3), Fr(1, 3), 0]
    paare13 = [frozenset({j}) for j in range(d)] + [frozenset(range(d)) - {j} for j in range(d)]
    ph = H.kernel_bruch(v, d, paare13, [1] * 8)
    assert ph == [Fr(3, 8), Fr(3, 8), Fr(3, 8), Fr(-1, 8)] and ph[3] != 0              # das unbeteiligte Merkmal bekommt −1/8
    assert H.kernel_bruch(v, d, inner, [1] * len(inner)) == [Fr(5, 16), Fr(5, 16), Fr(5, 16), Fr(1, 16)]   # alle Koalitionen ungewichtet: auch nicht exakt
    v2 = H.spiel(d, {(0,): Fr(1), (0, 1): Fr(2), (1, 2): Fr(-3)})
    assert H.shapley_koal(d, lambda S: v2[frozenset(S)]) == [Fr(2), Fr(-1, 2), Fr(-3, 2), Fr(0)] == H.kernel_bruch(v2, d, paare13, [1] * 8)   # Ordnung 2: Paare der Größen 1 und 3 genügen


# ---------------------------------------------------------------------------------------------- Teil 6: Möbius-Ordnung von Baumspielen
def _zufallsbaum(rnd, d, k, used=()):
    if k == 0 or len(used) == d:
        return ("L", Fr(rnd.randint(-6, 6)))
    j = rnd.choice([q for q in range(d) if q not in used])
    return ("N", j, 0, _zufallsbaum(rnd, d, k - 1, used + (j,)), _zufallsbaum(rnd, d, k - 1, used + (j,)), None, None)


def _massen(t, punkte):
    if t[0] == "L":
        return t
    _, j, thr, a, b, _, _ = t
    p0 = [(p, w) for p, w in punkte if p[j] <= thr]
    p1 = [(p, w) for p, w in punkte if p[j] > thr]
    return ("N", j, thr, _massen(a, p0), _massen(b, p1), sum(w for _, w in p0), sum(w for _, w in p1))


def _blatt(t, z):
    while t[0] == "N":
        t = t[3] if z[t[1]] <= t[2] else t[4]
    return t[1]


def test_teil6_ordnung_hoechstens_tiefe_in_brueche():
    d = 4
    rnd = random.Random(7)
    pts = list(itertools.product((0, 1), repeat=d))
    hinter = [(p, Fr(rnd.randint(1, 9))) for p in pts]                                  # abhängiger Hintergrund mit beliebigen positiven Gewichten
    tot = sum(w for _, w in hinter)
    erreicht = {1: 0, 2: 0, 3: 0}
    for k in (1, 2, 3):
        for _ in range(60):
            t = _massen(_zufallsbaum(rnd, d, k), hinter)
            x = rnd.choice(pts)
            vint = {frozenset(S): sum(w * _blatt(t, tuple(x[j] if j in S else b[j] for j in range(d))) for b, w in hinter) / tot
                    for r in range(d + 1) for S in itertools.combinations(range(d), r)}
            vpfad = {frozenset(S): H.alg1_tupel(t, x, frozenset(S)) for r in range(d + 1) for S in itertools.combinations(range(d), r)}
            for nm, v in (("interventional", vint), ("pfadabhängig", vpfad)):
                c = H.moebius_bruch(d, v)
                ordn = max([len(T_) for T_, val in c.items() if val != 0 and len(T_) > 0] or [0])
                assert ordn <= k, f"{nm}: Ordnung {ordn} größer als Tiefe {k}"
                erreicht[k] = max(erreicht[k], ordn)
                ex = H.shapley_koal(d, lambda S: v[frozenset(S)])
                assert ex == [sum((cv / len(T_) for T_, cv in c.items() if i in T_), Fr(0)) for i in range(d)]
                # Code der Demo (Gleitkomma) gegen den Bruch
                vec = [float(v[frozenset(i for i in range(d) if (kk >> i) & 1)]) for kk in range(1 << d)]
                assert MB.ordnung(MB.moebius(vec, d), d, 1e-9) == ordn
    assert erreicht == {1: 1, 2: 2, 3: 3}                                               # die Ordnung k wird bei Tiefe k erreicht
    kette = ("N", 0, 0, ("L", Fr(0)), ("N", 1, 0, ("L", Fr(0)), ("N", 2, 0, ("L", Fr(0)), ("L", Fr(1)), None, None), None, None), None, None)
    kt = _massen(kette, [(p, Fr(1)) for p in pts])
    v = {frozenset(S): H.alg1_tupel(kt, (1, 1, 1, 0), frozenset(S)) for r in range(d + 1) for S in itertools.combinations(range(d), r)}
    c = H.moebius_bruch(d, v)
    assert all(c[T_] == Fr(1, 8) for r in range(4) for T_ in itertools.combinations(range(3), r))     # UND-Kette: c_T = 1/8 für alle T in {0, 1, 2}
    assert all(c[T_] == 0 for T_ in c if 3 in T_) and max(len(T_) for T_, val in c.items() if val != 0) == 3
