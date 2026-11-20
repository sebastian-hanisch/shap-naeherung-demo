"""Eingefrorene Messungen laden (`data/*.json`, geschrieben von `tools/shn_make_*.py`; Quelle und Stand stehen im Feld `meta` jeder Datei).

Die App rechnet nur mit diesen Zahlen und den eigenen Rechenwegen (`shn_tree`, `shn_treeshap`, `shn_kernel`, `shn_moebius`); scikit-learn kommt zur Laufzeit nicht vor."""

import functools
import json
from pathlib import Path

DATA = Path(__file__).resolve().parent / "data"


@functools.lru_cache(maxsize=None)
def _lade(name: str) -> dict:
    return json.loads((DATA / name).read_text(encoding="utf-8"))


def learn(zweit: bool = False) -> dict:
    return _lade("shn_learn_zweit.json" if zweit else "shn_learn.json")


def costs(zweit: bool = False) -> dict:
    return _lade("shn_costs_zweit.json" if zweit else "shn_costs.json")


def games(d: int, zweit: bool = False) -> dict:
    return _lade(f"shn_games_d{d}{'_zweit' if zweit else ''}.json")


def rausch_text(rauschen: float) -> str:
    return "ohne Rauschen" if rauschen == 0 else "Rauschen 0,5"


def lern_label(zeile: dict, welt: str) -> str:
    """Anzeigename einer Zeile von Experiment 2 (Toy oder Vehikel)."""
    if welt == "toy":
        return f"Toy · {zeile['lernverfahren']} · n = {zeile['n']} · {rausch_text(zeile['rauschen'])}"
    return f"Vehikel (Ziel Vorlauf) · {zeile['lernverfahren']}"


def lern_zeilen(zweit: bool = False) -> dict:
    """Label → (Welt, Zeile) in der Reihenfolge der Datei (zuerst Toy, dann Vehikel)."""
    d = learn(zweit)
    out = {}
    for z in d["toy"]["zeilen"]:
        out[lern_label(z, "toy")] = ("toy", z)
    for z in d["vehikel"]["zeilen"]:
        out[lern_label(z, "vehikel")] = ("vehikel", z)
    return out


def spiel_label(d: int, modell: str) -> str:
    return f"d = {d} · {modell}"


def spiel_modelle(zweit: bool = False) -> dict:
    """Label → (d, Modelleintrag) für alle eingefrorenen Spieltabellen (d = 6, dann d = 10)."""
    out = {}
    for d in (6, 10):
        for m in games(d, zweit)["modelle"]:
            out[spiel_label(d, m["modell"])] = (d, m)
    return out
