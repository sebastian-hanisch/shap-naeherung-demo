"""Rechenweg 2 für das pfadabhängige Funktional: TS20 Algorithmus 2 (EXTEND/UNWIND, polynomiell in Blättern und Tiefe), mit Schrittzähler.

Quelle des Algorithmus: Lundberg u. a., „Explainable AI for Trees“ (arXiv, v1), Algorithmus 2 (Stufe W, gelesen). Diese Fassung ist eine eigene Implementierung nach
der Beschreibung; sie ist nur über Orakel abgesichert (Aufzählung, alle Permutationen, Brüche), nicht über das Paket `shap`.

Zählregel (PLAN-02b Befund D): ein **Schritt** ist ein Aufruf der Knotenrekursion plus je eine Schleifenrunde in EXTEND und in UNWIND. Kopien der Pfadliste und die
Summen über die Gewichte sind nicht gezählt. Die Zählregel der Aufzählung steht in `shn_tree.werte_aufzaehlung` (ein Schritt = ein Besuch von EXPVALUE).
Zahlen: float, oder `fractions.Fraction` (dann `eins=Fraction(1)`), dann ist das Ergebnis exakt."""

from shn_tree import Baum


def treeshap_poly(b: Baum, x, d: int, eins=1.0):
    """(φ (Liste der Länge d), Zahl der Schritte). `eins` legt die Zahlenart fest (1.0 oder Fraction(1))."""
    phi = [eins * 0 for _ in range(d)]
    zaehler = [0]

    def extend(m, pz, po, pi):
        l = len(m)
        m = [e[:] for e in m]
        m.append([pi, pz, po, eins if l == 0 else eins * 0])
        for i in range(l - 1, -1, -1):
            zaehler[0] += 1
            m[i + 1][3] += po * m[i][3] * (eins * (i + 1) / (l + 1))
            m[i][3] = pz * m[i][3] * (eins * (l - i) / (l + 1))
        return m

    def unwind(m, i):
        m = [e[:] for e in m]
        l = len(m) - 1
        n = m[l][3]
        for j in range(l - 1, -1, -1):
            zaehler[0] += 1
            if m[i][2] != 0:
                tmp = m[j][3]
                m[j][3] = n * (eins * (l + 1) / (j + 1)) / m[i][2]
                n = tmp - m[j][3] * m[i][1] * (eins * (l - j) / (l + 1))
            else:
                m[j][3] = m[j][3] * (eins * (l + 1)) / (m[i][1] * (l - j))
        w = [e[3] for e in m[:l]]
        rest = [e[:3] for k, e in enumerate(m) if k != i]
        return [[r[0], r[1], r[2], w[k]] for k, r in enumerate(rest)]

    def rec(knoten, m, pz, po, pi):
        zaehler[0] += 1
        m = extend(m, pz, po, pi)
        if b.left[knoten] == -1:
            for i in range(1, len(m)):
                w = sum(e[3] for e in unwind(m, i))
                phi[m[i][0]] += w * (m[i][2] - m[i][1]) * b.value[knoten]
            return
        j = b.feat[knoten]
        a, c = b.left[knoten], b.right[knoten]
        h, k = (a, c) if x[j] <= b.thr[knoten] else (c, a)
        iz = io = eins
        for q in range(1, len(m)):
            if m[q][0] == j:
                iz, io = m[q][1], m[q][2]
                m = unwind(m, q)
                break
        rec(h, m, iz * b.cover[h] / b.cover[knoten], io, j)
        rec(k, m, iz * b.cover[k] / b.cover[knoten], eins * 0, j)

    rec(0, [], eins, eins, -1)
    return phi, zaehler[0]


def schritte_vollstaendiger_baum(d: int) -> int:
    """Geschlossene Form für den vollständigen Baum der Tiefe d über d verschiedenen Merkmalen: Σ_{k=0..d} 2^k (1 + k) + 2^d d²."""
    return sum(2 ** k * (1 + k) for k in range(d + 1)) + 2 ** d * d * d


def besuche_vollstaendiger_baum(d: int) -> int:
    """Knotenbesuche der Aufzählung (alle 2^d Koalitionen) am vollständigen Baum der Tiefe d über d Merkmalen: 3^(d+1) − 2^(d+1)."""
    return 3 ** (d + 1) - 2 ** (d + 1)
