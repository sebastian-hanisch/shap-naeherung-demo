"""Shapley-Wert aus den Werten aller Koalitionen (Koalitionsformel) für Spiele mit d Spielern.

Koalitionen sind Bitmasken: Bit j gesetzt heißt „Merkmal j gehört zur Koalition“. `vals[k]` ist der Wert der Koalition k (2^d Werte).
Die Funktion rechnet mit Zahlen jeder Art (float, numpy-float, `fractions.Fraction`): mit Brüchen ist das Ergebnis exakt (Tests, Handbeispiele)."""

from fractions import Fraction
from math import factorial

import numpy as np


def gewicht(d: int, groesse: int) -> Fraction:
    """Shapley-Gewicht einer Koalition der Größe s ohne Spieler j: s! (d − s − 1)! / d!."""
    return Fraction(factorial(groesse) * factorial(d - groesse - 1), factorial(d))


def shapley_aus_werten(vals, d: int) -> list:
    """Shapley-Werte (Liste der Länge d) aus `vals` (2^d Werte). Bei Bruchwerten exakt."""
    if len(vals) != 1 << d:
        raise ValueError(f"{len(vals)} Werte, erwartet {1 << d}")
    out = []
    for j in range(d):
        s = 0
        for k in range(1 << d):
            if (k >> j) & 1:
                continue
            s += gewicht(d, bin(k).count("1")) * (vals[k | (1 << j)] - vals[k])
        out.append(s)
    return out


_GEW = {}


def _gewichte_np(d: int):
    """(Größen der Koalitionen, Gewichte als float) in Bitmaskenreihenfolge, für die schnelle Fassung."""
    if d not in _GEW:
        groessen = np.array([bin(k).count("1") for k in range(1 << d)])
        w = np.array([float(gewicht(d, s)) if s < d else 0.0 for s in range(d + 1)])
        _GEW[d] = (groessen, w)
    return _GEW[d]


def shapley_np(V, d: int):
    """Schnelle Fassung für float-Arrays: V (…, 2^d) → Shapley-Werte (…, d)."""
    V = np.asarray(V, dtype=float)
    groessen, w = _gewichte_np(d)
    phi = np.zeros(V.shape[:-1] + (d,))
    ks = np.arange(1 << d)
    for j in range(d):
        ohne = ks[((ks >> j) & 1) == 0]
        phi[..., j] = (w[groessen[ohne]] * (V[..., ohne | (1 << j)] - V[..., ohne])).sum(axis=-1)
    return phi
