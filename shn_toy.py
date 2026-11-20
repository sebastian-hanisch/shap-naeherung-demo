"""Binäres Toy (Handbeispiel-Welt, ausgedacht): drei Merkmale, x1 gleichverteilt, x2 = x1 mit Wahrscheinlichkeit q, x3 unabhängig und gleichverteilt; f = x1.

Acht Punkte, die Verteilung steht in einer Tabelle (`verteilung`), kein Zufall. Das Toy ist die Handbeispiel-Welt dieser Demo, kein zweites Vehikel der Reihe.
Hier stehen: die Aufzählung aller Binärbäume, die f auf allen acht Punkten berechnen (für q = 9/10 sind es 131), die Beiträge je Baum (pfadabhängig, Rechenweg 1)
und die beiden Wahrheiten der Funktion f je Funktional (interventional, konditional), jeweils aus der Verteilung, ohne Baum.

Merkmale hier 0-indiziert: x1 = Spalte 0, x2 = Spalte 1, x3 = Spalte 2."""

import itertools
from fractions import Fraction

import numpy as np

from shn_shapley import shapley_aus_werten
from shn_tree import Baum, werte_aufzaehlung

D = 3
PUNKTE = list(itertools.product((0, 1), repeat=D))
Q_STANDARD = Fraction(9, 10)


def verteilung(q=Q_STANDARD) -> dict:
    """Punkt → Wahrscheinlichkeit: x1 und x3 gleichverteilt, x2 = x1 mit Wahrscheinlichkeit q (q ein Bruch für exakte Tests, sonst float)."""
    return {p: (q if p[1] == p[0] else 1 - q) / 4 for p in PUNKTE}


def f(p) -> int:
    """Die Funktion, die alle Bäume berechnen: f = x1."""
    return p[0]


def baumformen(q=Q_STANDARD) -> list:
    """Alle Binärbäume (Schnitte x_j <= 0 auf beliebiger Reihenfolge, jedes Merkmal höchstens einmal je Pfad), die f auf allen acht Punkten berechnen, als `Baum`.
    Ein Knoten wird nur dort gesetzt, wo beide Seiten Punkte tragen; ein Blatt steht, wo f im Bereich konstant ist. Für q = 9/10 sind es 131 Bäume (1 / 10 / 120 nach Tiefe 1 / 2 / 3)."""
    pr = verteilung(q)

    def masse(region):
        return sum(pr[p] for p in region)

    def gen(region, tiefe_frei, benutzt):
        res = []
        if len({f(p) for p in region}) == 1:
            res.append(("L", Fraction(f(next(iter(region)))) if isinstance(q, Fraction) else float(f(next(iter(region))))))
        if tiefe_frei > 0:
            for j in range(D):
                if j in benutzt:
                    continue
                r0 = [p for p in region if p[j] == 0]
                r1 = [p for p in region if p[j] == 1]
                if r0 and r1:
                    for a in gen(r0, tiefe_frei - 1, benutzt | {j}):
                        for c in gen(r1, tiefe_frei - 1, benutzt | {j}):
                            res.append(("N", j, 0, a, c, masse(r0), masse(r1)))
        return res

    return [Baum.aus_tupel(t) for t in gen(PUNKTE, D, frozenset())]


def baum_beschreibung(b: Baum) -> str:
    """Kurzschrift eines Baums, z. B. `x2 ? (x1 ? 0 : 1) : (x1 ? 0 : 1)` wird als `x2[x1,x1]`; Blatt als Wert."""
    def rec(i):
        if b.left[i] == -1:
            return "0" if b.value[i] == 0 else "1"
        return f"x{b.feat[i] + 1}[{rec(b.left[i])},{rec(b.right[i])}]"
    return rec(0)


def tiefe(b: Baum) -> int:
    return b.tiefe()


def wurzel_merkmal(b: Baum):
    """Merkmal an der Wurzel (0-indiziert) oder None bei einem Baum nur aus einem Blatt."""
    return None if b.left[0] == -1 else b.feat[0]


def pfade_mit_masse(b: Baum):
    """Je Blatt (Merkmalsfolge des Pfads, Masse des Blatts als Anteil der Gesamtmasse)."""
    res = []

    def rec(i, pre, m):
        if b.left[i] == -1:
            res.append((tuple(pre), m))
            return
        tot = b.cover[b.left[i]] + b.cover[b.right[i]]
        rec(b.left[i], pre + [b.feat[i]], m * b.cover[b.left[i]] / tot)
        rec(b.right[i], pre + [b.feat[i]], m * b.cover[b.right[i]] / tot)

    rec(0, [], 1 if not isinstance(b.cover[0], float) else 1.0)
    return res


def masse_x2_vor_x1(b: Baum):
    """Masse der Blätter, auf deren Pfad x2 (Spalte 1) vor x1 (Spalte 0) getrennt wird (x2 steht im Pfad, und entweder x1 danach oder gar nicht)."""
    return sum((m for pf, m in pfade_mit_masse(b) if 1 in pf and (0 not in pf or pf.index(1) < pf.index(0))), 0)


def beitraege_pfad(b: Baum, q=Q_STANDARD) -> dict:
    """Pfadabhängige Beiträge φ des Baums an jedem der acht Punkte (Rechenweg 1, Aufzählung): Punkt → Liste der drei Werte."""
    return {x: shapley_aus_werten(werte_aufzaehlung(b, x, D)[0], D) for x in PUNKTE}


def wichtigkeit(phi_je_punkt: dict, q=Q_STANDARD) -> list:
    """Wichtigkeit je Merkmal: Mittel von |φ|, gewichtet mit der Verteilung."""
    pr = verteilung(q)
    return [sum(pr[x] * abs(phi_je_punkt[x][j]) for x in PUNKTE) for j in range(D)]


def wahrheit_interventional(q=Q_STANDARD) -> dict:
    """Beiträge der Funktion f (kein Baum) mit dem interventionalen Funktional: v(S) = Σ_b p(b) f(x_S, b_rest)."""
    pr = verteilung(q)
    out = {}
    for x in PUNKTE:
        def v(k, x=x):
            return sum(pr[b] * f(tuple(x[j] if (k >> j) & 1 else b[j] for j in range(D))) for b in PUNKTE)
        out[x] = shapley_aus_werten([v(k) for k in range(1 << D)], D)
    return out


def wahrheit_konditional(q=Q_STANDARD) -> dict:
    """Beiträge der Funktion f (kein Baum) mit dem konditionalen Funktional: v(S) = E[f(X) | X_S = x_S]."""
    pr = verteilung(q)
    out = {}
    for x in PUNKTE:
        def v(k, x=x):
            sel = [b for b in PUNKTE if all(b[j] == x[j] for j in range(D) if (k >> j) & 1)]
            return sum(pr[b] * f(b) for b in sel) / sum(pr[b] for b in sel)
        out[x] = shapley_aus_werten([v(k) for k in range(1 << D)], D)
    return out


def klassen_zaehlen(wichtigkeiten, nachkommastellen=9):
    """Gruppiert die Wichtigkeiten (Liste von Dreiergruppen) nach ihrem Wert auf `nachkommastellen` Stellen: Liste (Wichtigkeit, Zahl der Bäume), aufsteigend nach dem Beitrag von x2."""
    zaehl = {}
    for w in wichtigkeiten:
        key = tuple(round(float(v), nachkommastellen) for v in w)
        zaehl[key] = zaehl.get(key, 0) + 1
    return sorted(zaehl.items(), key=lambda kv: (kv[0][1], kv[0][2]))


def gleitkomma_beitraege(baeume, q: float):
    """Wichtigkeit (float) je Baum bei der Verteilung q: Liste von Listen mit drei Werten. Schnell genug für alle 131 Bäume (Millisekunden)."""
    return [wichtigkeit(beitraege_pfad(b, q), q) for b in baeume]


def modell_tabelle(b: Baum) -> np.ndarray:
    """Vorhersage des Baums an den acht Punkten (Reihenfolge von PUNKTE)."""
    return np.array([b.vorhersage(p) for p in PUNKTE], dtype=float)
