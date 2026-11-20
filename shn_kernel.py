"""KernelSHAP (eigene Implementierung, kein Paket): Koalitionen ziehen, ungewichtete kleinste Quadrate mit der Nebenbedingung Σφ = v(N) − v(∅).

* **Ziehen:** die Größe s ∈ {1, …, d−1} mit Wahrscheinlichkeit ∝ (d−1)/(s (d−s)) (Shapley-Kern, Lundberg/Lee 2017), dann eine zufällige Teilmenge dieser Größe;
  Koalitionen werden mit Zurücklegen gezogen (doppelte kommen vor). **Einfach:** M Ziehungen. **Gepaart:** M/2 Ziehungen, jede zusammen mit ihrem Komplement (M Koalitionen).
* **Lösen:** die Koalitionen gehen mit gleichem Gewicht in die Regression ein (die Größenverteilung des Ziehens trägt den Shapley-Kern); die letzte Unbekannte wird über die
  Nebenbedingung eliminiert. Mit **allen** Koalitionen und dem Shapley-Gewicht ist das Ergebnis exakt (LL17 Satz 2): `kernel_voll_gewichtet` (Orakel).
* **Singulär:** spannen die gezogenen Koalitionen den Raum nicht auf (Rang < d − 1), ist die Lösung nicht eindeutig; `kernel_loesen` meldet das, statt es zu verschweigen.
Zufall: SplitMix64 (`shn_rng`), auf jeder Plattform bitgleich."""

from math import comb

import numpy as np

from shn_rng import SplitMix64


def groessen_wahrscheinlichkeit(d: int):
    """Wahrscheinlichkeit der Koalitionsgrößen 1..d−1 beim Ziehen nach dem Shapley-Kern."""
    s = np.arange(1, d)
    p = (d - 1) / (s * (d - s))
    return list(p / p.sum())


def masken_ziehen(d: int, m_koalitionen: int, rng: SplitMix64, gepaart: bool):
    """Gezogene Koalitionen als bool-Array (M, d). Gepaart: M/2 Ziehungen und ihre Komplemente (M gerade)."""
    if d < 2:
        raise ValueError("KernelSHAP braucht mindestens d = 2 Merkmale")
    if gepaart and m_koalitionen % 2:
        raise ValueError("gepaart braucht eine gerade Zahl M")
    n_zieh = m_koalitionen // 2 if gepaart else m_koalitionen
    p = groessen_wahrscheinlichkeit(d)
    mk = np.zeros((n_zieh, d), dtype=bool)
    for i in range(n_zieh):
        s = rng.choice(p) + 1
        reihe = list(range(d))
        rng.shuffle(reihe)
        mk[i, reihe[:s]] = True
    return np.vstack([mk, ~mk]) if gepaart else mk


def masken_index(masken):
    """Bitmasken-Index der Koalitionen (Bit j = Merkmal j)."""
    masken = np.asarray(masken, dtype=bool)
    return (masken.astype(np.int64) * (1 << np.arange(masken.shape[1]))).sum(axis=1)


def kernel_loesen(masken, v, v0: float, v_voll: float, gewichte=None):
    """(φ̂ (d,), singulär: bool). `v` sind die Werte der gezogenen Koalitionen. Ohne `gewichte` zählt jede Koalition gleich.
    Die Lösung ist bei Singularität die mit kleinster Norm (lstsq) und trotzdem nicht eindeutig; `singulär` zeigt das an."""
    Z = np.asarray(masken, dtype=float)
    d = Z.shape[1]
    delta = v_voll - v0
    y = np.asarray(v, dtype=float) - v0 - Z[:, -1] * delta
    A = Z[:, :-1] - Z[:, [-1]]
    w = np.ones(len(Z)) if gewichte is None else np.sqrt(np.asarray(gewichte, dtype=float))
    Aw = A * w[:, None]
    sol = np.linalg.lstsq(Aw, y * w, rcond=None)[0]
    singulaer = int(np.linalg.matrix_rank(Aw)) < d - 1
    return np.append(sol, delta - sol.sum()), bool(singulaer)


def kernel_voll_gewichtet(vals, d: int):
    """Orakel (LL17 Satz 2): alle inneren Koalitionen mit Shapley-Gewicht (d−1)/(C(d,s) s (d−s)): ergibt den exakten Shapley-Wert (bis auf Rundung)."""
    ks = np.arange(1, (1 << d) - 1)
    groessen = np.array([bin(k).count("1") for k in ks])
    w = np.array([(d - 1) / (comb(d, s) * s * (d - s)) for s in groessen])
    masken = ((ks[:, None] >> np.arange(d)) & 1).astype(bool)
    vals = np.asarray(vals, dtype=float)
    phi, _ = kernel_loesen(masken, vals[ks], vals[0], vals[-1], w)
    return phi


def kosten_auswertungen(m_koalitionen: int, hintergrund: int) -> int:
    """Modellauswertungen: M Koalitionen mal Hintergrundzeilen."""
    return m_koalitionen * hintergrund
