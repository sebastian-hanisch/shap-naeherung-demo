"""Entscheidungsbaum als Felder (wie scikit-learn: left, right, feat, thr, value, cover) und das pfadabhängige Wertfunktional des Baums.

* Knoten i ist ein Blatt, wenn `left[i] == -1`; sonst geht `x[feat[i]] <= thr[i]` nach links. `cover[i]` ist die Masse des Knotens (Trainingsanteile), `value[i]` der Blattwert.
* Rechenweg 1 (Aufzählung, TS20 Algorithmus 1, EXPVALUE): `expvalue` rechnet v(S) für eine Koalition S; Merkmale in S folgen dem Punkt, die anderen mischen
  die beiden Kinder nach ihrer Masse. `werte_aufzaehlung` tut das für alle 2^d Koalitionen und zählt die Knotenbesuche.
* `blatt_rechnung` ist ein zweiter, vektorisierter Rechenweg für dasselbe v(S) (über Blattanteile), er dient als Orakel und rechnet große Ensembles.
Zahlen können float oder `fractions.Fraction` sein (Tests rechnen mit Brüchen). Alles hier ist Rechnung am Baum, nicht an einer Funktion: dasselbe f gibt je Baumform andere v(S)."""

from dataclasses import dataclass

import numpy as np


@dataclass
class Baum:
    left: list
    right: list
    feat: list
    thr: list
    value: list
    cover: list

    @property
    def n_knoten(self) -> int:
        return len(self.left)

    def ist_blatt(self, i: int) -> bool:
        return self.left[i] == -1

    def n_blaetter(self) -> int:
        return sum(1 for i in range(self.n_knoten) if self.left[i] == -1)

    def tiefe(self) -> int:
        def rec(i):
            return 0 if self.left[i] == -1 else 1 + max(rec(self.left[i]), rec(self.right[i]))
        return rec(0)

    def vorhersage(self, x) -> float:
        i = 0
        while self.left[i] != -1:
            i = self.left[i] if x[self.feat[i]] <= self.thr[i] else self.right[i]
        return self.value[i]

    def als_dict(self) -> dict:
        return {k: list(getattr(self, k)) for k in ("left", "right", "feat", "thr", "value", "cover")}

    @classmethod
    def aus_dict(cls, d: dict) -> "Baum":
        return cls(*(list(d[k]) for k in ("left", "right", "feat", "thr", "value", "cover")))

    @classmethod
    def aus_tupel(cls, t) -> "Baum":
        """Baum aus verschachtelten Tupeln: ("L", Wert) oder ("N", Merkmal, Schwelle, links, rechts, Masse_links, Masse_rechts)."""
        b = cls([], [], [], [], [], [])

        def neu(t, masse):
            i = len(b.left)
            for lst, v in ((b.left, -1), (b.right, -1), (b.feat, -1), (b.thr, 0), (b.value, 0), (b.cover, masse)):
                lst.append(v)
            if t[0] == "L":
                b.value[i] = t[1]
                return i
            _, j, thr, a, c, ma, mc = t
            b.feat[i], b.thr[i] = j, thr
            b.left[i] = neu(a, ma)
            b.right[i] = neu(c, mc)
            return i

        neu(t, t[5] + t[6] if t[0] == "N" else 1)
        return b


def expvalue(b: Baum, x, maske: int, knoten: int = 0, zaehler=None):
    """Wert v(S) der Koalition S (Bitmaske) am Punkt x: TS20 Algorithmus 1. `zaehler` (Liste mit einer Zahl) zählt die Knotenbesuche."""
    if zaehler is not None:
        zaehler[0] += 1
    if b.left[knoten] == -1:
        return b.value[knoten]
    j = b.feat[knoten]
    a, c = b.left[knoten], b.right[knoten]
    if (maske >> j) & 1:
        return expvalue(b, x, maske, a if x[j] <= b.thr[knoten] else c, zaehler)
    return (b.cover[a] * expvalue(b, x, maske, a, zaehler) + b.cover[c] * expvalue(b, x, maske, c, zaehler)) / b.cover[knoten]


def werte_aufzaehlung(b: Baum, x, d: int):
    """(v(S) für alle 2^d Koalitionen in Bitmaskenreihenfolge, Zahl der Knotenbesuche über alle Koalitionen). Zählregel Rechenweg 1: ein Schritt = ein Besuch EXPVALUE."""
    z = [0]
    vals = [expvalue(b, x, k, 0, z) for k in range(1 << d)]
    return vals, z[0]


def blatt_pfade(b: Baum):
    """Je Blatt (Wert, [(Merkmal, Schwelle, geht_links, Massenverhältnis des Kindes zum Elternknoten), …]) entlang des Pfads von der Wurzel."""
    res = []

    def rec(i, pfad):
        if b.left[i] == -1:
            res.append((b.value[i], pfad))
            return
        j, thr = b.feat[i], b.thr[i]
        a, c = b.left[i], b.right[i]
        rec(a, pfad + [(j, thr, True, b.cover[a] / b.cover[i])])
        rec(c, pfad + [(j, thr, False, b.cover[c] / b.cover[i])])

    rec(0, [])
    return res


def blatt_rechnung(b: Baum, X, d: int):
    """Zweiter Rechenweg für v(S) (Orakel): Summe über die Blätter von Blattwert mal Produkt je Merkmal (folgt x, falls das Merkmal in S ist, sonst Massenanteil).
    X (n, d) → (n, 2^d) in Bitmaskenreihenfolge; nur float."""
    X = np.asarray(X, dtype=float)
    blaetter = blatt_pfade(b)
    n, L = len(X), len(blaetter)
    R = np.ones((L, d))
    A = np.ones((n, L, d))
    val = np.array([v for v, _ in blaetter], dtype=float)
    for li, (_, pfad) in enumerate(blaetter):
        for f, thr, links, ratio in pfad:
            R[li, f] *= ratio
            A[:, li, f] *= ((X[:, f] <= thr) == links)
    out = np.zeros((n, 1 << d))
    for k in range(1 << d):
        in_s = np.array([(k >> j) & 1 for j in range(d)], dtype=bool)
        W = np.where(in_s[None, None, :], A, R[None, :, :]).prod(axis=2)
        out[:, k] = W @ val
    return out
