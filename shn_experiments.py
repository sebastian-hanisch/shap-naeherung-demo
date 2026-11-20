"""Die Rechnungen der fünf Experimente (rein, ohne Streamlit): jede Funktion bekommt die Reglerwerte und die eingefrorenen Daten und gibt Zahlen zurück.
Die Texte der Verdikte stehen in `shn_evaluation.py`.

Zähleinheiten: Baumformen, Zufalls-Seeds (Trainingsdatensätze), Instanzen (Zeilen einer Instanzmenge), Koalitionen, Modellauswertungen (Koalitionen mal Hintergrundzeilen),
Zufalls-Seeds der Ziehung (Ziehungen), Spiele, Schritte. Keine Sekunden als Befund."""

import functools
from fractions import Fraction

import numpy as np

import shn_data as DATA
import shn_toy as T
from shn_kernel import kernel_loesen, kernel_voll_gewichtet, masken_index, masken_ziehen
from shn_moebius import anteile_je_ordnung, moebius, ordnung, zufallsspiel
from shn_rng import MASK, SplitMix64
from shn_shapley import shapley_aus_werten, shapley_np
from shn_tree import Baum, werte_aufzaehlung
from shn_treeshap import treeshap_poly


def seed_aus(*zahlen: int) -> int:
    """Ein Zufalls-Seed aus mehreren ganzen Zahlen (feste Mischung, auf jeder Plattform gleich)."""
    s = 0x9E3779B97F4A7C15
    for z in zahlen:
        s = ((s ^ (int(z) + 0x632BE59BD9B4E019)) * 0xBF58476D1CE4E5B9) & MASK
        s ^= s >> 31
    return s


# ======================================================================================================= Experiment 1: eine Funktion, viele Bäume
FUNKTIONALE = ("pfadabhängig", "interventional", "konditional")


@functools.lru_cache(maxsize=None)
def formen(q: float):
    """Alle 131 Baumformen für f = x1 bei der Verteilung q (als `Baum`), in fester Reihenfolge (Tiefe, Wurzel, Kurzschrift)."""
    bs = T.baumformen(q)
    return sorted(bs, key=lambda b: (b.tiefe(), -1 if T.wurzel_merkmal(b) is None else T.wurzel_merkmal(b), T.baum_beschreibung(b)))


def form_label(b: Baum) -> str:
    w = T.wurzel_merkmal(b)
    return f"Tiefe {b.tiefe()} · Wurzel x{w + 1} · {T.baum_beschreibung(b)}" if w is not None else "ein Blatt"


@functools.lru_cache(maxsize=None)
def form_labels() -> tuple:
    """Die Namen der Baumformen (hängen nicht von q ab, solange 0 < q < 1)."""
    return tuple(form_label(b) for b in formen(0.9))


def _modell_funktional(tab, q: float, funktional: str):
    """Wichtigkeit (Mittel von |φ|, mit der Verteilung gewichtet) des **Baummodells** (Tabelle seiner Vorhersagen an den acht Punkten) bei interventionalem oder konditionalem Funktional."""
    pr = np.array([float(T.verteilung(q)[p]) for p in T.PUNKTE])
    P8 = np.array(T.PUNKTE, float)
    code = np.array([4, 2, 1])
    masken = np.array([[(k >> j) & 1 for j in range(3)] for k in range(8)], dtype=bool)
    V = np.zeros((8, 8))
    for xi, x in enumerate(P8):
        for k in range(8):
            if funktional == "interventional":
                Z = np.where(masken[k][None, :], x[None, :], P8)
                V[xi, k] = pr @ tab[(Z @ code).astype(int)]
            else:
                sel = np.all(P8[:, masken[k]] == x[masken[k]][None, :], axis=1)
                V[xi, k] = (pr[sel] @ tab[sel]) / pr[sel].sum()
    return pr @ np.abs(shapley_np(V, 3))


@functools.lru_cache(maxsize=None)
def wichtigkeiten_pfad(q: float) -> tuple:
    """Wichtigkeit (x1, x2, x3) der pfadabhängigen Beiträge für alle 131 Baumformen (Tupel von Dreiergruppen)."""
    return tuple(tuple(w) for w in T.gleitkomma_beitraege(formen(q), q))


@functools.lru_cache(maxsize=None)
def wahrheiten(q: float) -> dict:
    """Wahrheit der Funktion f = x1 je Funktional, aus der Verteilung (kein Baum): Wichtigkeit je Merkmal."""
    return {"interventional": [float(v) for v in T.wichtigkeit(T.wahrheit_interventional(q), q)],
            "konditional": [float(v) for v in T.wichtigkeit(T.wahrheit_konditional(q), q)]}


def exp1(q: float, form_index: int, funktional: str) -> dict:
    bs = formen(q)
    b = bs[form_index]
    if funktional == "pfadabhängig":
        alle = [tuple(w) for w in wichtigkeiten_pfad(q)]
    else:
        key = "interventional" if funktional == "interventional" else "konditional"
        alle = [tuple(_modell_funktional(T.modell_tabelle(t), q, key)) for t in bs]
    gewaehlt = alle[form_index]
    return {"q": q, "funktional": funktional, "n_formen": len(bs), "klassen": T.klassen_zaehlen(alle), "label": form_label(b), "tiefe": b.tiefe(),
            "wurzel": T.wurzel_merkmal(b), "wichtigkeit": list(gewaehlt), "masse_x2_vor_x1": float(T.masse_x2_vor_x1(b)),
            "wahrheit": wahrheiten(q), "alle_x2": [w[1] for w in alle],
            "modell_gleich_f": all(abs(b.vorhersage(p) - T.f(p)) < 1e-12 for p in T.PUNKTE)}


def mini_beispiel() -> dict:
    """Das Mini-Beispiel von Hand (zwei Merkmale, Verteilung (0,0) 9/20, (1,1) 9/20, (0,1) 1/20, (1,0) 1/20; f = x1; Punkt x = (1, 1)): zwei Bäume, derselbe Funktionswert,
    andere Beiträge. Exakt in Brüchen."""
    pr = {(0, 0): Fraction(9, 20), (1, 1): Fraction(9, 20), (0, 1): Fraction(1, 20), (1, 0): Fraction(1, 20)}
    L0, L1 = ("L", Fraction(0)), ("L", Fraction(1))
    A = Baum.aus_tupel(("N", 0, 0, L0, L1, Fraction(1, 2), Fraction(1, 2)))
    B = Baum.aus_tupel(("N", 1, 0, ("N", 0, 0, L0, L1, pr[(0, 0)], pr[(1, 0)]), ("N", 0, 0, L0, L1, pr[(0, 1)], pr[(1, 1)]), Fraction(1, 2), Fraction(1, 2)))
    x = (1, 1)
    out = {}
    for nm, b in (("A", A), ("B", B)):
        phi = shapley_aus_werten(werte_aufzaehlung(b, x, 2)[0], 2)
        imp = [sum(pr[p] * abs(shapley_aus_werten(werte_aufzaehlung(b, p, 2)[0], 2)[j]) for p in pr) for j in range(2)]
        out[nm] = {"phi": phi, "wichtigkeit": imp}
    return out


# ======================================================================================================= Experiment 2: Lernverfahren
def exp2(label: str) -> dict:
    welt, z = DATA.lern_zeilen()[label]
    if welt == "toy":
        w = DATA.learn()["toy"]["wahrheit"]
        return {"welt": "toy", "zeile": z, "wahrheit": w, "luecke_x2": z["pfad"][1] - z["interv"][1]}
    v = DATA.learn()["vehikel"]
    art_i = [m[0] for m in z["interv_je_menge"]]
    art_p = [m[0] for m in z["pfad_je_menge"]]
    luecke = [p - i for p, i in zip(art_p, art_i)]
    return {"welt": "vehikel", "zeile": z, "wahrheit": v["wahrheit_je_menge"], "art_interv": (min(art_i), max(art_i)), "art_pfad": (min(art_p), max(art_p)),
            "luecke_art": (min(luecke), max(luecke)), "n_mengen": len(art_i)}


# ======================================================================================================= Experiment 3: zwei Rechenwege
BAUM_ARTEN = ("trainiert, Tiefe 3", "trainiert, Tiefe 6", "trainiert, Tiefe 8", "vollständig (Tiefe = d)")


def voll_baum(d: int, ebene: int = 0):
    """Vollständiger Baum der Tiefe d über d verschiedenen Merkmalen (Merkmal k auf Ebene k), gleiche Massen: Tupelform für `Baum.aus_tupel`."""
    if ebene == d:
        return ("L", float(ebene))
    k = float(2 ** (d - ebene - 1))
    return ("N", ebene, 0, voll_baum(d, ebene + 1), voll_baum(d, ebene + 1), k, k)


def exp3(baum_art: str, d: int) -> dict:
    """Beide Rechenwege live an einem Baum (eingefroren aus scikit-learn oder vollständig) und einem Punkt; Zähler laut Zählregel."""
    if baum_art == "vollständig (Tiefe = d)":
        b = Baum.aus_tupel(voll_baum(d))
        x = (1.0,) * d
        eingefroren = None
    else:
        tiefe = int(baum_art.split("Tiefe ")[1])
        eintrag = next(e for e in DATA.costs()["kleine_baeume"] if e["d"] == d and e["tiefe"] == tiefe)
        b, x, eingefroren = Baum.aus_dict(eintrag["baum"]), eintrag["x"], eintrag
    vals, besuche = werte_aufzaehlung(b, x, d)
    phi1 = shapley_np(np.array(vals)[None, :], d)[0]
    phi2, schritte = treeshap_poly(b, x, d)
    return {"d": d, "baum_art": baum_art, "blaetter": b.n_blaetter(), "tiefe": b.tiefe(), "besuche": besuche, "schritte": schritte,
            "abweichung": float(np.abs(phi1 - np.array(phi2)).max()), "effizienz": float(abs(sum(phi2) - (b.vorhersage(x) - vals[0]))),
            "phi": [float(v) for v in phi2], "eingefroren": (eingefroren["besuche"], eingefroren["schritte"]) if eingefroren else None}


def kosten_tabelle() -> list:
    """Zeilen der eingefrorenen Kostentabelle: d, Tiefe, Blätter (min, max), Aufzählung (Mittel, min, max), Algorithmus 2 (Mittel, min, max), Verhältnis (min, max), Bäume mit mehr Schritten."""
    out = []
    for z in DATA.costs()["tabelle"]:
        c1, c2 = np.array(z["aufzaehlung"]), np.array(z["alg2"])
        r = c1 / c2
        out.append({"d": z["d"], "tiefe": z["tiefe"], "n_baeume": z["n_baeume"], "blaetter": (min(z["blaetter"]), max(z["blaetter"])),
                    "aufzaehlung": (float(c1.mean()), int(c1.min()), int(c1.max())), "alg2": (float(c2.mean()), int(c2.min()), int(c2.max())),
                    "verhaeltnis": (float(r.min()), float(r.max())), "alg2_mehr": int((c2 > c1).sum()), "abweichung": z["abweichung_max"]})
    return out


def ensembles() -> list:
    """Abstand pfadabhängig gegen interventional am trainierten Ensemble: je Modell Spannen über die Instanzmengen."""
    out = []
    for z in DATA.costs()["ensembles"]["zeilen"]:
        out.append({"modell": z["modell"], "r2": z["r2_gegen_mu"], "median": (min(z["median_je_menge"]), max(z["median_je_menge"])),
                    "q90": (min(z["q90_je_menge"]), max(z["q90_je_menge"])), "rang1": (min(z["rang1_anders_je_menge"]), max(z["rang1_anders_je_menge"])),
                    "vorzeichen": (min(z["vorzeichen_anders_je_menge"]), max(z["vorzeichen_anders_je_menge"]))})
    return out


# ======================================================================================================= Experiment 4: KernelSHAP
def kernel_experiment(d: int, modell: str, m_koal: int, gepaart: bool, ziehungen: int, zweit: bool = False) -> dict:
    """KernelSHAP gegen die Aufzählung auf den eingefrorenen Spieltabellen. Je Instanzmenge (fünf getrennte) und je Ziehung werden M Koalitionen gezogen (für alle Zeilen der Menge
    dieselben, Zufall aus SplitMix64) und für jede Zeile das Regressionsproblem gelöst. Fehler: größter |φ̂ − φ| geteilt durch das größte |φ| der Aufzählung."""
    g = DATA.games(d, zweit)
    eintrag = next(m for m in g["modelle"] if m["modell"] == modell)
    V = np.array(eintrag["v"])                                   # (Mengen, Zeilen, 2^d)
    exakt = shapley_np(V, d)                                     # (Mengen, Zeilen, d)
    fehler_je_menge, rang_je_menge, alle_fehler, singulaer, n_ziehungen = [], [], [], 0, 0
    for s in range(V.shape[0]):
        fe, rg = [], []
        for z in range(ziehungen):
            rng = SplitMix64(seed_aus(d, s, z, m_koal, int(gepaart)))
            mk = masken_ziehen(d, m_koal, rng, gepaart)
            idx = masken_index(mk)
            n_ziehungen += 1
            for i in range(V.shape[1]):
                phi, sing = kernel_loesen(mk, V[s, i][idx], V[s, i][0], V[s, i][-1])
                if i == 0:
                    singulaer += sing           # der Rang hängt nur an den gezogenen Koalitionen, nicht an der Zeile: eine Zählung je Ziehung
                fe.append(float(np.abs(phi - exakt[s, i]).max() / np.abs(exakt[s, i]).max()))
                rg.append(bool(np.argmax(np.abs(phi)) == np.argmax(np.abs(exakt[s, i]))))
        fehler_je_menge.append(float(np.median(fe)))
        rang_je_menge.append(float(np.mean(rg)))
        alle_fehler.extend(fe)
    n_hint = g["hintergrund"]
    return {"d": d, "modell": modell, "m": m_koal, "gepaart": gepaart, "ziehungen": ziehungen, "n_mengen": V.shape[0], "zeilen_je_menge": V.shape[1],
            "median_je_menge": fehler_je_menge, "rang1_je_menge": rang_je_menge, "q90": float(np.quantile(alle_fehler, 0.9)), "singulaer": singulaer, "n_ziehungen": n_ziehungen,
            "auswertungen": m_koal * n_hint, "auswertungen_aufzaehlung": (1 << d) * n_hint, "anteil": m_koal / (1 << d), "r2_gegen_mu": eintrag["r2_gegen_mu"]}


def orakel_alle_koalitionen(d: int, modell: str, zweit: bool = False) -> float:
    """Größte Abweichung von KernelSHAP mit allen Koalitionen und Shapley-Gewicht gegen die Aufzählung (Orakel, LL17 Satz 2) über alle Zeilen aller Mengen eines Modells."""
    eintrag = next(m for m in DATA.games(d, zweit)["modelle"] if m["modell"] == modell)
    V = np.array(eintrag["v"])
    exakt = shapley_np(V, d)
    return float(max(np.abs(kernel_voll_gewichtet(V[s, i], d) - exakt[s, i]).max() for s in range(V.shape[0]) for i in range(min(2, V.shape[1]))))


# ======================================================================================================= Experiment 5: die Ordnung entscheidet
def ordnung_tabelle(zweit: bool = False) -> list:
    """Je Modell (d = 6): größte Möbius-Ordnung und Anteil der Quadratsumme je Ordnung (aus den Spieltabellen neu gerechnet, Schwelle 1e-9), die eingefrorenen Werte zur Kontrolle."""
    out = []
    for m in DATA.games(6, zweit)["modelle"]:
        c = moebius(np.array(m["v"]), 6)
        out.append({"modell": m["modell"], "ordnung": int(ordnung(c, 6, 1e-9)), "anteile": anteile_je_ordnung(c, 6).tolist(),
                    "ordnung_eingefroren": m["ordnung_max"], "anteile_eingefroren": m["anteile_je_ordnung"]})
    return out


@functools.lru_cache(maxsize=None)
def _zufallsspiele(d: int, ordn: int, gruppe: int, n: int = 40, salz: int = 0):
    """40 Zufallsspiele einer Gruppe samt exakten Shapley-Werten (unabhängig von m, darum zwischengespeichert)."""
    spiele = [zufallsspiel(d, ordn, SplitMix64(seed_aus(d, ordn, gruppe, t, 1, salz))) for t in range(n)]
    V = np.array(spiele)
    return V, shapley_np(V, d)


def zufallsspiele(d: int, ordn: int, m_paare: int, gruppen: int = 5, n: int = 40, salz: int = 0) -> dict:
    """Anteil der Zufallsspiele (Möbius-Koeffizienten N(0, 1) bis Ordnung `ordn`), bei denen gepaartes Ziehen mit `m_paare` Paaren den Shapley-Wert exakt trifft
    (größter Fehler < 1e-9), je Gruppe zu `n` Spielen. Ziehverteilung: Größen aus dem Shapley-Kern. Singuläre Ziehungen zählen als nicht exakt und werden mitgezählt."""
    anteile, sing = [], 0
    for g in range(gruppen):
        V, exakt = _zufallsspiele(d, ordn, g, n, salz)
        ok = 0
        for t in range(n):
            rng = SplitMix64(seed_aus(d, ordn, m_paare, t, g, 2, salz))
            mk = masken_ziehen(d, 2 * m_paare, rng, True)
            idx = masken_index(mk)
            phi, s = kernel_loesen(mk, V[t][idx], V[t][0], V[t][-1])
            sing += s
            ok += bool(np.abs(phi - exakt[t]).max() < 1e-9)
        anteile.append(ok / n)
    return {"d": d, "ordnung": ordn, "m": m_paare, "n_spiele": n, "anteile": anteile, "singulaer": sing, "ziehungen": gruppen * n}


def gepaart_fehler_je_modell(ziehungen: int = 10, m_koal: int = 24, zweit: bool = False) -> dict:
    """Je Modell mit Spieltabelle bei d = 6: (kleinster, größter) Medianfehler über die fünf Instanzmengen für gepaartes Ziehen mit `m_koal` Koalitionen."""
    out = {}
    for m in DATA.games(6, zweit)["modelle"]:
        r = kernel_experiment(6, m["modell"], m_koal, True, ziehungen, zweit)
        out[m["modell"]] = (min(r["median_je_menge"]), max(r["median_je_menge"]))
    return out
