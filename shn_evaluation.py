"""Verdikte als reine Funktionen: jede liefert (Art, Text), die App ruft `getattr(st, Art)(Text)`. `tests/test_verdicts.py` prüft Schwellen und jede Zahl im Text an handgesetzten Werten.

Zahlen im Text haben das Dezimalkomma (`de`). Jede Zahl nennt ihre Zähleinheit (Baumformen, Zufalls-Seeds, Instanzen, Koalitionen, Modellauswertungen, Schritte) und ihre Voraussetzung."""

import numpy as np

import shn_experiments as E


def de(x: float, nd: int = 3) -> str:
    """Zahl mit Dezimalkomma und `nd` Nachkommastellen."""
    return f"{x:.{nd}f}".replace(".", ",")


def tausend(n: int) -> str:
    """Ganze Zahl mit Leerzeichen als Tausendertrenner (geschütztes Leerzeichen)."""
    return f"{int(n):,}".replace(",", " ")


def spanne(lo: float, hi: float, nd: int = 3) -> str:
    return de(lo, nd) if de(lo, nd) == de(hi, nd) else f"{de(lo, nd)} bis {de(hi, nd)}"


def prozent(x: float, nd: int = 1) -> str:
    return de(100 * x, nd) + " %"


# ------------------------------------------------------------------------------------------------------- Experiment 1
def verdict_formen(r: dict) -> tuple[str, str]:
    """Eine Funktion, viele Bäume. `r` ist das Ergebnis von `shn_experiments.exp1`."""
    n = r["n_formen"]
    klassen = r["klassen"]
    w2 = r["wichtigkeit"][1]
    if r["funktional"] != "pfadabhängig":
        return "info", (f"{r['funktional'].capitalize()}: alle {n} Baumformen geben denselben Beitrag (x₂: {de(w2)}), denn dieses Funktional liest die Vorhersagen des Baums, "
                        "nicht seine Form; nur das pfadabhängige Funktional hängt an der Baumform.")
    if len(klassen) == 1:
        return "info", (f"Bei q = {de(r['q'], 2)} geben alle {n} Baumformen dieselben Beiträge: x₂ ist hier von x₁ unabhängig, und alle Funktionale stimmen überein.")
    kond = r["wahrheit"]["konditional"][1]
    teile = []
    for (w, anzahl) in klassen:
        if abs(w[1]) < 1e-9:
            teile.append(f"{anzahl} dem nie gelesenen x₂ den Beitrag 0")
        elif abs(w[1] - kond) < 1e-6:
            teile.append(f"{anzahl} den konditionalen Wert {de(w[1])}")
        else:
            teile.append(f"{anzahl} den Wert {de(w[1])}")
    text = f"f = x₁: von {n} Baumformen geben " + ", ".join(teile) + f" (Verteilung q = {de(r['q'], 2)}). Gewählt: x₂ bekommt hier {de(w2)}"
    if abs(r["q"] - 0.9) < 1e-9:
        text += f"; der Beitrag ist ein Fünftel der Masse der Blätter, auf deren Pfad x₂ vor x₁ getrennt wird (hier {de(r['masse_x2_vor_x1'], 2)} · 1/5 = {de(r['masse_x2_vor_x1'] / 5)})."
    else:
        text += "."
    if r["wurzel"] == 0:
        text += " Liegt x₁ an der Wurzel, bekommt x₂ immer 0."
    return ("warning" if w2 > 1e-9 else "info"), text


# ------------------------------------------------------------------------------------------------------- Experiment 2
def verdict_lernen(r: dict) -> tuple[str, str]:
    """Was macht ein Lernverfahren daraus? `r` ist das Ergebnis von `shn_experiments.exp2`."""
    z = r["zeile"]
    if r["welt"] == "toy":
        p2, i2, n_seeds = z["pfad"][1], z["interv"][1], z["n_seeds"]
        k = int(round(z["anteil_phi2"] * n_seeds))
        wk = r["wahrheit"]["konditional"][1]
        wahr = f"Wahrheit der Funktion f: interventional 0, konditional {de(wk)}."
        if z["rauschen"] == 0:
            gleich = "gleich f" if z["abweichung_von_f"] < 1e-9 else f"von f um höchstens {de(z['abweichung_von_f'], 4)} entfernt"
            text = (f"Dieselbe Funktion (Modellwerte auf den acht Punkten {gleich}): für x₂ interventional {de(i2)}, pfadabhängig {de(p2)} "
                    f"(Mittel über {n_seeds} Zufalls-Seeds; |φ₂| > 0,02 in {k} von {n_seeds}). {wahr}")
            return ("warning" if p2 > 0.02 else "success"), text
        text = (f"Mit Rauschen 0,5 ist das Modell nicht mehr f: für x₂ interventional {de(i2)}, pfadabhängig {de(p2)} (Mittel über {n_seeds} Zufalls-Seeds; "
                f"|φ₂| > 0,02 in {k} von {n_seeds}). Das Modell liest x₂ hier wirklich (interventional ungleich 0). {wahr}")
        return "info", text
    lo_i, hi_i = r["art_interv"]
    lo_p, hi_p = r["art_pfad"]
    lo_l, hi_l = r["luecke_art"]
    kond = [m[0] for m in r["wahrheit"]["konditional"]]
    kopf = f"Ziel g = Vorlauf (liest nur den Vorlauf), {z['lernverfahren']} (R² {de(z['r2'], 4)} gegen g): Beitrag der Art "
    if hi_i < 0.005 and hi_p < 0.005:
        return "success", kopf + f"interventional {spanne(lo_i, hi_i)} und pfadabhängig {spanne(lo_p, hi_p)} (fünf Instanzmengen zu je 25 Zeilen, 5 Zufalls-Seeds): das Modell liest die Art nicht; Wahrheit von g: Art interventional 0, konditional {spanne(min(kond), max(kond), 2)}."
    return "warning", (kopf + f"interventional {spanne(lo_i, hi_i)}, pfadabhängig {spanne(lo_p, hi_p)} (fünf Instanzmengen zu je 25 Zeilen, 5 Zufalls-Seeds): das Modell liest die Art wirklich, "
                       f"die Lücke zwischen den Funktionalen ist {spanne(lo_l, hi_l, 2)}; Wahrheit von g: Art interventional 0, konditional {spanne(min(kond), max(kond), 2)}.")


# ------------------------------------------------------------------------------------------------------- Experiment 3
def verdict_kosten_live(r: dict) -> tuple[str, str]:
    """Beide Rechenwege an dem gewählten Baum: Zähler, Verhältnis, Übereinstimmung der Beiträge."""
    v, s = r["besuche"], r["schritte"]
    kopf = f"Dieser Baum (d = {r['d']}, Tiefe {r['tiefe']}, {r['blaetter']} Blätter, ein Punkt): Aufzählung {tausend(v)} Besuche, Algorithmus 2 {tausend(s)} Schritte"
    gleich = f"; beide geben dieselben Beiträge (größte Abweichung {r['abweichung']:.1e})".replace(".", ",")
    if s > v:
        return "warning", kopf + f", also {de(s / v, 2)}-mal so viele wie die Aufzählung" + gleich + "."
    return "info", kopf + f", also {de(v / s, 2)}-mal weniger" + gleich + "."


def verdict_kosten_tabelle(zeilen: list) -> tuple[str, str]:
    """Anker aus der eingefrorenen Kostentabelle: d = 14 mit Tiefe 3 gegen d = 6 mit Tiefe 8."""
    a = next(z for z in zeilen if z["d"] == 14 and z["tiefe"] == 3)
    b = next(z for z in zeilen if z["d"] == 6 and z["tiefe"] == 8)
    return "info", (f"d = 14, Tiefe 3: Aufzählung {tausend(a['aufzaehlung'][0])} Schritte, Algorithmus 2 {tausend(a['alg2'][1])} bis {tausend(a['alg2'][2])} ({a['n_baeume']} Bäume); "
                    f"bei d = 6 und Tiefe 8 braucht Algorithmus 2 in {b['alg2_mehr']} von {b['n_baeume']} Bäumen mehr ({tausend(b['alg2'][1])} bis {tausend(b['alg2'][2])} gegen "
                    f"{tausend(b['aufzaehlung'][1])} bis {tausend(b['aufzaehlung'][2])}; Erstmessung, in der Zweitmessung mit anderen Zufalls-Seeds in 6 von 8). Die zwei Einheiten sind nicht gleich teuer: die Zählung zeigt das Wachstum, nicht den Preis.")


def verdict_ensembles(zeilen: list) -> tuple[str, str]:
    """Abstand pfadabhängig gegen interventional am trainierten Ensemble (Spannen über fünf Instanzmengen)."""
    teile = [f"{z['modell']}: Median {spanne(100 * z['median'][0], 100 * z['median'][1], 1)} %, anderes Rang 1 in {spanne(100 * z['rang1'][0], 100 * z['rang1'][1], 1)} % der Zeilen"
             for z in zeilen]
    return "info", ("Relativer Abstand zwischen pfadabhängigem und interventionalem Beitrag desselben Modells (Summe der Beträge der Unterschiede geteilt durch die Summe der interventionalen Beträge; "
                    "fünf Instanzmengen zu je 30 Zeilen, 5 Zufalls-Seeds): " + "; ".join(teile) + ".")


# ------------------------------------------------------------------------------------------------------- Experiment 4
def verdict_kernel(r: dict) -> tuple[str, str]:
    """KernelSHAP: Kosten gegen Fehler. `r` ist das Ergebnis von `shn_experiments.kernel_experiment`."""
    lo, hi = min(r["median_je_menge"]), max(r["median_je_menge"])
    rlo, rhi = min(r["rang1_je_menge"]), max(r["rang1_je_menge"])
    variante = "gepaart" if r["gepaart"] else "einfach"
    text = (f"{r['modell']}, {variante}, M = {r['m']} Koalitionen = {tausend(r['auswertungen'])} von {tausend(r['auswertungen_aufzaehlung'])} Modellauswertungen der Aufzählung "
            f"({prozent(r['anteil'])}): Medianfehler je Instanzmenge {spanne(lo, hi)} ({r['n_mengen']} Mengen zu {r['zeilen_je_menge']} Zeilen, je {r['ziehungen']} Ziehungen), "
            f"90 %-Quantil {de(r['q90'])}; gleiches Rang 1 wie die Aufzählung in {spanne(100 * rlo, 100 * rhi, 0)} % der Läufe.")
    if r["singulaer"]:
        text += f" {r['singulaer']} von {r['n_ziehungen']} Ziehungen waren singulär (die gezogenen Koalitionen spannen den Raum nicht auf; die Lösung ist dort nicht eindeutig)."
    if r["anteil"] > 1.0:
        return "warning", text + " Diese Ziehung ist teurer als die Aufzählung, wenn jede Ziehung neu ausgewertet wird (doppelte Ziehungen brauchen nur eine Auswertung)."
    if hi < 1e-6:
        return "success", text + " Der Fehler ist null bis auf Rundung: das Spiel dieses Modells hat höchstens Wechselwirkungen zweiter Ordnung."
    return ("info" if hi <= 0.05 else "warning"), text


# ------------------------------------------------------------------------------------------------------- Experiment 5
def verdict_ordnung(zeilen: list, fehler: dict) -> tuple[str, str]:
    """Reihe der Boosting-Modelle: Ordnung des Spiels und gepaarter Fehler bei 24 Koalitionen (d = 6). `fehler`: Modell → (kleinster, größter Medianfehler über die Mengen)."""
    teile = []
    for k in (1, 2, 3, 4):
        nm = f"Boosting Tiefe {k}"
        o = next(z for z in zeilen if z["modell"] == nm)["ordnung"]
        lo, hi = fehler[nm]
        teile.append(f"Tiefe {k}: Ordnung {o}, Fehler {'0,000' if hi < 5e-4 else spanne(lo, hi)}")
    return "info", ("Gepaartes KernelSHAP bei 24 von 64 Koalitionen, Boosting (100 Bäume), d = 6, Medianfehler je Instanzmenge (fünf Mengen zu 10 Zeilen): " + "; ".join(teile)
                    + ". Die Ordnung des Spiels ist hier höchstens die Baumtiefe (gemessen an diesen Modellen, kein Gesetz).")


def verdict_zufallsspiele(r: dict) -> tuple[str, str]:
    """Anteil exakter Zufallsspiele. `r` ist das Ergebnis von `shn_experiments.zufallsspiele`."""
    d, o, m, n = r["d"], r["ordnung"], r["m"], r["n_spiele"]
    ok = [round(a * n) for a in r["anteile"]]
    gesamt = len(ok) * n
    if m < d - 1:
        return "warning", (f"Mit {m} Paaren (weniger als d − 1 = {d - 1}) sind die Normalgleichungen singulär: es gibt keinen eindeutigen Wert, nur eine Meldung statt einer Zahl "
                           f"({r['singulaer']} von {r['ziehungen']} Ziehungen singulär; exakt in {sum(ok)} von {gesamt} Spielen der Ordnung {o}, d = {d}).")
    text = (f"Zufallsspiele der Ordnung {o}, d = {d}, {m} gezogene Paare ({2 * m} Koalitionen): exakt in {spanne(100 * min(r['anteile']), 100 * max(r['anteile']), 0)} % von je {n} Spielen "
            f"({len(ok)} Gruppen, zusammen {sum(ok)} von {gesamt} Spielen).")
    if o >= 3:
        return ("warning", text + (" In keinem dieser Spiele exakt: bei Ordnung 3 verwechselt die Regression den kubischen Term mit linearen." if sum(ok) == 0
                                   else " Bei Ordnung 3 ist gepaartes Ziehen nicht allgemein exakt."))
    return ("success" if min(r["anteile"]) >= 0.95 else "info"), text
