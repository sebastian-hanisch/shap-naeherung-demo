"""Möbius-Koeffizienten eines Spiels: v(S) = Σ_{T ⊆ S} c_T. Die **Ordnung** des Spiels ist die größte Größe |T| mit c_T ≠ 0.

Es gilt φ_i = Σ_{T ∋ i} c_T / |T| (zweite Definition desselben Shapley-Werts, als Orakel). Spiele sind Vektoren der Länge 2^d in Bitmaskenreihenfolge.
Zufallsspiele (`zufallsspiel`) haben Koeffizienten N(0, 1) bis zu einer gewählten Ordnung (SplitMix64, `shn_rng`)."""

import math

import numpy as np

from shn_rng import SplitMix64

POPCOUNT_CACHE = {}


def popcount(d: int):
    if d not in POPCOUNT_CACHE:
        POPCOUNT_CACHE[d] = np.array([bin(k).count("1") for k in range(1 << d)])
    return POPCOUNT_CACHE[d]


def moebius(v, d: int):
    """Möbius-Koeffizienten (Teilmengen-Transformation): c[k] für die Koalition k; Form (…, 2^d)."""
    c = np.array(v, dtype=float)
    ks = np.arange(1 << d)
    for j in range(d):
        mit = ks[((ks >> j) & 1) == 1]
        c[..., mit] -= c[..., mit ^ (1 << j)]
    return c


def zeta(c, d: int):
    """Umkehrung: v(S) = Σ_{T ⊆ S} c_T."""
    v = np.array(c, dtype=float)
    ks = np.arange(1 << d)
    for j in range(d):
        mit = ks[((ks >> j) & 1) == 1]
        v[..., mit] += v[..., mit ^ (1 << j)]
    return v


def ordnung(c, d: int, schwelle: float = 1e-9) -> int:
    """Größte Teilmengengröße (ohne die leere Menge) mit |c_T| > schwelle; 0, wenn es keine gibt."""
    c = np.asarray(c, dtype=float)
    pc = popcount(d)
    groessen = [o for o in range(1, d + 1) if np.abs(c[..., pc == o]).max() > schwelle]
    return max(groessen) if groessen else 0


def anteile_je_ordnung(c, d: int):
    """Anteil der Quadratsumme der Koeffizienten (ohne den konstanten Term) je Ordnung 1..d; Summe 1 (oder 0, wenn alle Koeffizienten 0 sind)."""
    c = np.asarray(c, dtype=float)
    pc = popcount(d)
    q = np.array([(c[..., pc == o] ** 2).sum() for o in range(1, d + 1)])
    return q / q.sum() if q.sum() > 0 else q


def shapley_aus_moebius(c, d: int):
    """φ_i = Σ_{T ∋ i} c_T / |T| (zweiter Rechenweg gegen die Koalitionsformel)."""
    c = np.asarray(c, dtype=float)
    pc = popcount(d)
    phi = np.zeros(d)
    for k in range(1, 1 << d):
        for i in range(d):
            if (k >> i) & 1:
                phi[i] += c[k] / pc[k]
    return phi


def zufallsspiel(d: int, order: int, rng: SplitMix64):
    """Spiel mit Möbius-Koeffizienten N(0, 1) für alle Mengen der Größe 1..order (der konstante Term ist 0); Vektor der Länge 2^d."""
    pc = popcount(d)
    c = np.zeros(1 << d)
    for k in range(1, 1 << d):
        if pc[k] <= order:
            # Box-Muller mit math.log/math.cos statt rng.normal(): 150 µs je Aufruf wären bei 200 Spielen mit d = 10 zu langsam. Die Gleichverteilten bleiben bitgleich;
            # die Koeffizienten können sich unter Linux in der letzten Binärstelle unterscheiden, was die Unterscheidung exakt (Fehler ~1e-15) gegen nicht exakt (~1e-1) nicht kippt.
            u1, u2 = max(rng.random(), 1e-12), rng.random()
            c[k] = math.sqrt(-2.0 * math.log(u1)) * math.cos(2.0 * math.pi * u2)
    return zeta(c, d)
