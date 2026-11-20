"""Erfundener Container-Generator (Verweilzeit): KOPIE der Teile aus `verweilzeit-demo` (`vwz_generator.py`, `vwz_constants.py`), die hier gebraucht werden.

Alles hier ist ausgedacht (kein Echtdatenbezug). Kopiert, nicht importiert: `tests/test_generator_kopie.py` prüft die Kopie gegen eingefrorene Werte der Quelle
(Kundeneffekte, Orakel-Mitte, die ersten Zeilen einer Historie). Die wahre mittlere Log-Verweilzeit mu(x) ist analytisch bekannt (`log_median_and_s`).

Merkmale (Spalten): 0 art (0 Import, 1 Export, 2 Umschlag, 3 Leer), 1 kunde (0..11), 2 zoll, 3 wochenende, 4 vorlauf, 5 gewicht (reines Rauschen).
Zollhalt gibt es nur bei Import, Vorlauf nur bei Export und Umschlag: daraus entsteht die Abhängigkeit zwischen den Merkmalen, an der sich der Rechenweg des Baum-SHAP verrät."""

import numpy as np

ART_LABELS = ("Import", "Export", "Umschlag", "Leer")
ART_P = (0.35, 0.30, 0.25, 0.10)             # Anteile der Containerarten
K_KUNDEN = 12                                # Kunden mit je einem Effekt auf die log-Verweilzeit
KUNDEN_SEED = 12345
U_SD = 0.30                                  # Streuung der Kundeneffekte (log)
U_SCALE = (1.0, 0.3, 0.3, 1.0)               # Kundeneffekt je Art
S_BASE = (0.55, 0.25, 0.35, 0.90)            # Streuung der log-Verweilzeit je Art
P_ZOLL = 0.25                                # Zollhalt bei Import
HISTORY_SEED_OFFSET = 10_000_000             # Trainingshistorien haben einen eigenen Zufallsstrom-Bereich
NAMEN = ("Art", "Kunde", "Zoll", "Wochenende", "Vorlauf", "Gewicht")


def kunden_effekte(k_seed: int = KUNDEN_SEED):
    return np.random.default_rng(k_seed).normal(0.0, U_SD, K_KUNDEN)


UK = kunden_effekte()


def log_median_and_s(art, kunde, zoll, we, vorlauf, noise_scale=1.0):
    """Analytische bedingte Verteilung: log-Verweilzeit ~ N(mu, s^2). Vektorisiert. Das ist die „bekannte Funktion μ“."""
    art = np.asarray(art)
    u = UK[kunde] * np.take(U_SCALE, art)
    med = np.select(
        [art == 0, art == 1, art == 2],
        [3.5 * np.exp(0.7 * zoll + 0.30 * we), 0.6 + 0.85 * vorlauf, 0.3 + 0.8 * vorlauf],
        default=7.0 * np.exp(0.30 * we),
    )
    mu = np.log(med) + u
    s = np.take(S_BASE, art) * noise_scale
    return mu, s


def draw_features(rng, n):
    """Unabhängige Historie ohne Blockbeschränkung (Ankunftszeit gleichverteilt über viele Wochen); Spalten wie in der Quelle, plus eine Nullspalte am Ende."""
    art = rng.choice(4, size=n, p=ART_P)
    kunde = rng.integers(0, K_KUNDEN, n)
    zoll = ((rng.random(n) < P_ZOLL) & (art == 0)).astype(int)
    tday = rng.random(n) * 7000.0
    we = ((np.floor(tday).astype(int) % 7) >= 5).astype(int)
    vorlauf = np.where(art == 1, rng.uniform(1, 10, n), np.where(art == 2, rng.uniform(0.5, 6, n), 0.0))
    gewicht = rng.normal(0, 1, n)
    return np.column_stack([art, kunde, zoll, we, vorlauf, gewicht, np.zeros(n)]).astype(float)


def make_history(n, seed, noise_scale=1.0):
    """n abgeschlossene Container aus einer eigenen Historie: Merkmale (7 Spalten, die letzte ungenutzt) und Verweilzeit in Tagen."""
    rng = np.random.default_rng(HISTORY_SEED_OFFSET + seed)
    X = draw_features(rng, n)
    mu, s = log_median_and_s(X[:, 0].astype(int), X[:, 1].astype(int), X[:, 2], X[:, 3], X[:, 4], noise_scale)
    z = rng.standard_normal(n)
    za = rng.standard_normal(n)  # in der Quelle für die Ankündigung gezogen; hier ungenutzt, aber gezogen, damit der Strom gleich bleibt
    del za
    d = np.exp(mu + s * z)
    return X, d


def daten(n, seed, noise_scale=1.0):
    """(Merkmale (n, 6), log-Verweilzeit (n,)) der Historie `seed`."""
    X, d = make_history(n, seed, noise_scale)
    return X[:, :6].copy(), np.log(d)


def wahrheit(X):
    """Die bekannte Funktion μ(x) (wahre bedingte Mitte der Log-Verweilzeit) auf den Spalten 0 bis 4 von X."""
    mu, _ = log_median_and_s(X[:, 0].astype(int), X[:, 1].astype(int), X[:, 2], X[:, 3], X[:, 4])
    return mu
