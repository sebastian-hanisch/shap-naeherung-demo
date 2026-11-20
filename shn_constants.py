"""Konstanten: Regler-Grenzen, Voreinstellungen, Farben und Presets.

Die Auswahllisten (Lernverfahren, Spieltabellen) kommen aus den eingefrorenen Dateien in `data/` (`shn_data`); Zähleinheiten und Regeln stehen in `shn_experiments`."""

import shn_data as DATA

# --- Experiment 1: eine Funktion, viele Bäume -------------------------------------------------------------
Q_MIN, Q_MAX, DEFAULT_Q, Q_STEP = 0.50, 0.99, 0.90, 0.01
FUNKTIONALE = ("pfadabhängig", "interventional", "konditional")
DEFAULT_FUNKTIONAL = FUNKTIONALE[0]
FORM_NUR_X1 = "Tiefe 1 · Wurzel x1 · x1[0,1]"
FORM_X2_ZUERST = "Tiefe 2 · Wurzel x2 · x2[x1[0,1],x1[0,1]]"
DEFAULT_FORM = FORM_NUR_X1

# --- Experiment 2: Lernverfahren ---------------------------------------------------------------------------
LERN_OPTIONEN = tuple(DATA.lern_zeilen())
DEFAULT_LERN = "Toy · gieriger Baum, Tiefe 3 · n = 400 · ohne Rauschen"
LERN_WALD = "Toy · Wald 60 Bäume, 1 Merkmal je Schnitt · n = 400 · ohne Rauschen"

# --- Experiment 3: zwei Rechenwege --------------------------------------------------------------------------
BAUM_ARTEN = ("trainiert, Tiefe 3", "trainiert, Tiefe 6", "trainiert, Tiefe 8", "vollständig (Tiefe = d)")
DEFAULT_BAUM = BAUM_ARTEN[0]
D3_MIN, D3_MAX, DEFAULT_D3 = 4, 10, 6

# --- Experimente 4 und 5: KernelSHAP und Ordnung -------------------------------------------------------------
SPIEL_OPTIONEN = tuple(DATA.spiel_modelle())
DEFAULT_SPIEL = "d = 6 · Boosting Tiefe 3"
M_OPTIONEN = (12, 24, 40, 48, 80, 96, 160, 192, 320, 384, 640, 1280)
DEFAULT_M = 24
VARIANTEN = ("einfach", "gepaart")
DEFAULT_VARIANTE = "gepaart"
ZIEH_MIN, ZIEH_MAX, DEFAULT_ZIEH = 1, 100, 10
ORDNUNGEN = (1, 2, 3)
DEFAULT_ORDNUNG = 2
PAARE_MIN, PAARE_MAX, DEFAULT_PAARE = 1, 24, 8
SPIEL_D_OPTIONEN = (6, 10)
DEFAULT_SPIEL_D = 6

# Rechenschwelle: ab so vielen Löser-Aufrufen mal Koalitionen zeigt die App einen Spinner mit Zeitangabe (grobe Schätzung)
SPINNER_AB = 150_000

COLOR_PFAD = "#C0504D"
COLOR_INTERV = "#2F6B65"
COLOR_KOND = "#4c78a8"
COLOR_ALG1 = "#8c6d31"
COLOR_ALG2 = "#2F6B65"
COLOR_GRAU = "rgba(150,150,150,0.55)"

PRESETS = {
    "f = x₁, Baum nur x₁": {},
    "f = x₁, Baum x₂ zuerst": {"form": FORM_X2_ZUERST},
    "Wald mit zufälliger Merkmalswahl": {"lern": LERN_WALD},
    "Boosting Tiefe 2 (gepaart exakt)": {"spiel": "d = 6 · Boosting Tiefe 2"},
    "Boosting Tiefe 4": {"spiel": "d = 6 · Boosting Tiefe 4"},
    "Netz, d = 10": {"spiel": "d = 10 · Netz", "m": 40, "zieh": 6},
    "vollständiger Baum, d = 10": {"baum": "vollständig (Tiefe = d)", "d3": 10},
}
PRESET_HELP = {
    "f = x₁, Baum nur x₁": "Dieselbe Funktion f = x₁ in der einfachsten Baumform: nur x₁ wird gelesen, x₂ (zu 90 % gleich x₁) bekommt den Beitrag 0.",
    "f = x₁, Baum x₂ zuerst": "Dieselbe Funktion in einer Baumform, die zuerst auf x₂ trennt: x₂ bekommt jetzt den Beitrag 1/5, obwohl der Baum es für f gar nicht bräuchte.",
    "Wald mit zufälliger Merkmalswahl": "Ein Wald, der pro Schnitt nur ein zufälliges Merkmal sieht, berechnet auf den acht Punkten f bis auf höchstens 0,0167; sein pfadabhängiger Beitrag für x₂ ist trotzdem etwa 0,1, der interventionale etwa 0.",
    "Boosting Tiefe 2 (gepaart exakt)": "Boosting mit Tiefe 2 hat ein Spiel der Ordnung 2: gepaartes KernelSHAP ist mit 24 von 64 Koalitionen exakt, solange die Ziehung nicht singulär ist (bei d = 6 und 24 Koalitionen gab es in der Messung keine singuläre).",
    "Boosting Tiefe 4": "Boosting mit Tiefe 4 hat ein Spiel der Ordnung 4: derselbe Aufwand von 24 Koalitionen bleibt nicht mehr exakt.",
    "Netz, d = 10": "Zehn Merkmale (vier davon Rauschen), ein kleines Netz: 40 gepaarte Koalitionen kosten 3,9 % der Aufzählung von 1 024.",
    "vollständiger Baum, d = 10": "Der vollständige Baum über zehn Merkmalen: die Aufzählung besucht 175 099 Knoten, Algorithmus 2 braucht 122 881 Schritte (Zählregel in der Anzeige).",
}
