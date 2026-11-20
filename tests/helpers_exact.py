"""Exakte Orakel für die Tests (Brüche, nur Standardbibliothek): sie rechnen **nicht** mit den Modulen der Demo, sondern auf eigene Weise.

* `shapley_koal`: Koalitionsformel über Frozensets; `shapley_perm`: Mittel der Zuwächse über alle Permutationen (zweiter Weg desselben Werts).
* `alg1_tupel`: der Baumwert v(S) (TS20 Algorithmus 1) auf verschachtelten Tupeln, ohne `shn_tree`.
* `alle_baeume`: Aufzählung aller 131 Binärbäume für f = x1 (Handbeispiel-Welt) auf Tupeln, ohne `shn_toy`.
* `kernel_bruch`: KernelSHAP mit Gauß-Jordan in Brüchen (Elimination des letzten φ); wirft ZeroDivisionError bei singulärer Matrix.
* `moebius_bruch`: Möbius-Koeffizienten durch Einschluss-Ausschluss.
Tupelbäume: ("L", Wert) oder ("N", Merkmal, Schwelle, links, rechts, Masse_links, Masse_rechts); x[j] <= Schwelle geht nach links."""

import itertools
from fractions import Fraction as Fr
from math import comb, factorial


def shapley_koal(d, v):
    out = []
    for j in range(d):
        rest = [q for q in range(d) if q != j]
        s = Fr(0)
        for r in range(d):
            for S in itertools.combinations(rest, r):
                S = frozenset(S)
                s += Fr(factorial(r) * factorial(d - r - 1), factorial(d)) * (v(S | {j}) - v(S))
        out.append(s)
    return out


def shapley_perm(d, v):
    phi = [Fr(0)] * d
    perms = list(itertools.permutations(range(d)))
    for pm in perms:
        cur = frozenset()
        for j in pm:
            phi[j] += v(cur | {j}) - v(cur)
            cur = cur | {j}
    return [x / len(perms) for x in phi]


def alg1_tupel(t, x, S):
    if t[0] == "L":
        return t[1]
    _, j, thr, a, b, ca, cb = t
    if j in S:
        return alg1_tupel(a if x[j] <= thr else b, x, S)
    return (ca * alg1_tupel(a, x, S) + cb * alg1_tupel(b, x, S)) / (ca + cb)


D3 = 3
PTS = list(itertools.product((0, 1), repeat=D3))
PR = {p: Fr(1, 2) * (Fr(9, 10) if p[1] == p[0] else Fr(1, 10)) * Fr(1, 2) for p in PTS}


def alle_baeume():
    def cover(region):
        return sum(PR[p] for p in region)

    def gen(region, d):
        res = []
        if len({p[0] for p in region}) == 1:
            res.append(("L", Fr(next(iter(region))[0])))
        if d > 0:
            for j in range(D3):
                r0 = [p for p in region if p[j] == 0]
                r1 = [p for p in region if p[j] == 1]
                if r0 and r1:
                    for a in gen(r0, d - 1):
                        for b in gen(r1, d - 1):
                            res.append(("N", j, 0, a, b, cover(r0), cover(r1)))
        return res

    return gen(PTS, 3)


def blatt_pfade(t, pre=(), m=Fr(1)):
    if t[0] == "L":
        return [(pre, m)]
    _, j, _, a, b, ca, cb = t
    tot = ca + cb
    return blatt_pfade(a, pre + (j,), m * ca / tot) + blatt_pfade(b, pre + (j,), m * cb / tot)


def tiefe(t):
    return 0 if t[0] == "L" else 1 + max(tiefe(t[3]), tiefe(t[4]))


def spiel(d, coef):
    """Spiel aus Möbius-Koeffizienten: v(S) = Σ_{T ⊆ S} c_T (coef: Tupel → Bruch)."""
    return {S: sum((c for T, c in coef.items() if set(T) <= S), Fr(0)) for S in (frozenset(c) for r in range(d + 1) for c in itertools.combinations(range(d), r))}


def loese(M, b):
    n = len(M)
    M = [r[:] for r in M]
    b = b[:]
    for i in range(n):
        p = next((r for r in range(i, n) if M[r][i] != 0), None)
        if p is None:
            raise ZeroDivisionError("singulär")
        M[i], M[p] = M[p], M[i]
        b[i], b[p] = b[p], b[i]
        for r in range(n):
            if r != i and M[r][i] != 0:
                f = M[r][i] / M[i][i]
                M[r] = [M[r][c] - f * M[i][c] for c in range(n)]
                b[r] -= f * b[i]
    return [b[i] / M[i][i] for i in range(n)]


def kernel_bruch(v, d, koalitionen, gewichte):
    """KernelSHAP in Brüchen: gewichtete kleinste Quadrate mit der Nebenbedingung Σφ = v(N) − v(∅)."""
    v0, vN = v[frozenset()], v[frozenset(range(d))]
    delta = vN - v0
    rows = [[Fr(int(j in S) - int(d - 1 in S)) for j in range(d - 1)] for S in koalitionen]
    rhs = [v[S] - v0 - int(d - 1 in S) * delta for S in koalitionen]
    n = d - 1
    M = [[sum(Fr(w) * rows[k][i] * rows[k][j] for k, w in enumerate(gewichte)) for j in range(n)] for i in range(n)]
    bb = [sum(Fr(w) * rows[k][i] * rhs[k] for k, w in enumerate(gewichte)) for i in range(n)]
    sol = loese(M, bb)
    return sol + [delta - sum(sol)]


def moebius_bruch(d, v):
    out = {}
    for r in range(d + 1):
        for T in itertools.combinations(range(d), r):
            out[T] = sum(((-1) ** (r - len(U))) * v[frozenset(U)] for q in range(r + 1) for U in itertools.combinations(T, q))
    return out


def vollstaendiger_baum(d, level=0):
    if level == d:
        return ("L", Fr(level))
    a = vollstaendiger_baum(d, level + 1)
    b = vollstaendiger_baum(d, level + 1)
    return ("N", level, 0, a, b, Fr(2 ** (d - level - 1)), Fr(2 ** (d - level - 1)))
