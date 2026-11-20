"""Anzeigetexte der App (und die Textstücke, die die README wörtlich wiederholt: `tests/test_texts.py` vergleicht beide).

Alle Zahlen hier stehen in der README-Tabelle „Zahl → Test“ und in `tests/test_claims.py`. Alles Erfundene ist als erfunden gekennzeichnet; es gibt keine reale Datenquelle."""

TITEL = "Wie wird der Beitrag ausgerechnet? Baum-SHAP und KernelSHAP"

EINLEITUNG = """
Ein **Beitrag** (Shapley-Wert, hier φ) sagt, wie viel ein Merkmal zur Vorhersage einer einzelnen Zeile beiträgt. „SHAP“ heißt die Software und ihre Näherungen; der **Rechenweg**
bringt seine eigene Struktur mit: **Baum-SHAP** rechnet auf dem **Baum**, nicht auf der Funktion, und **KernelSHAP** zieht Koalitionen von Merkmalen und löst eine Regression: seine Genauigkeit
hängt an der **Ordnung der Wechselwirkungen**. Beides wird hier mit eigenen Rechenwegen nachgezählt, an einem erfundenen Container-Beispiel (Verweilzeit) und einem binären Handbeispiel (Toy).

**Drei Kernfragen**, jede mit Zahlen unten: (1) Rechnet Baum-SHAP den Beitrag der Funktion oder den des Baums? (2) Was kostet es, den Beitrag auszurechnen? (3) Wann ist die Näherung exakt, wann nicht?
Gezählt werden Baumformen, Zufalls-Seeds, Instanzen, Koalitionen, Modellauswertungen und Schritte, **keine Sekunden**. Es gibt keine reale Datenquelle.
Weiter unten: fünf Experimente, **🚧 Wo die Annahmen enden** und der Expander **📐 Mathematische Formulierung**.
"""

SO_FUNKTIONIERT = """
1. **Ein Spiel je Zeile.** Die Spieler sind die Merkmale. v(S) ist der Wert einer Koalition S von Merkmalen (was das Modell vorhersagt, wenn nur die Merkmale in S bekannt sind). Der **Shapley-Wert** φ
   verteilt v(alle) − v(leer) auf die Merkmale: jedes bekommt den gewichteten Mittelwert seiner Zuwächse v(S plus dieses Merkmal) − v(S) über alle Koalitionen S ohne es.
2. **Wertfunktionale.** Wie v(S) aus dem Modell entsteht, ist eine Wahl: **interventional** (die Merkmale außerhalb S werden aus einem Hintergrund eingesetzt), **konditional** (man bedingt auf die Merkmale in S;
   abhängige Merkmale sprechen dann mit) und **pfadabhängig** (ein Baum teilt für unbekannte Merkmale nach den Trainingsanteilen seiner Zweige). Das pfadabhängige Funktional gehört dem **Baum**, nicht der Funktion.
3. **Zwei Rechenwege für den Baum.** Die **Aufzählung** (Rechenweg 1) rechnet v(S) für alle 2^d Koalitionen. **Algorithmus 2** (Rechenweg 2) geht einmal durch den Baum und führt dabei die Gewichte der Pfade mit.
4. **KernelSHAP** zieht Koalitionen, rechnet für sie das Modell mit eingesetztem Hintergrund und löst eine Regression. **Gepaart** heißt: jede gezogene Koalition kommt mit ihrem Komplement.
   Gepaart gezogen ist es exakt, wenn die Ziehung nicht singulär ist (mindestens d − 1 verschiedene Paare) und das Spiel höchstens Wechselwirkungen zweiter Ordnung hat (Experiment 5).
"""

KERNFRAGE_1 = "Rechnet Baum-SHAP den Beitrag der Funktion oder den des Baums?"
KERNFRAGE_2 = "Was kostet es, den Beitrag auszurechnen?"
KERNFRAGE_3 = "Wann ist die Näherung exakt, wann nicht?"

EXP1_CAPTION = (
    "Zähleinheit: Baumformen (alle 131 Binärbäume, die f = x₁ auf allen acht Punkten berechnen; Brüche im Test, Gleitkomma live). Beitrag = Mittel von |φ|, gewichtet mit der Verteilung. "
    "Dieselbe Funktion, verschiedene Bäume: nur das pfadabhängige Funktional ändert sich mit der Form."
)
EXP2_CAPTION = (
    "Zähleinheit: Zufalls-Seeds = Trainingsdatensätze (30 im Toy, 5 im Vehikel); eingefroren (`tools/shn_make_learn.py`, scikit-learn-Modelle). Wahrheit „je Funktional“: die bekannte Funktion, nicht das Modell."
)
EXP3_CAPTION = (
    "Zählregel Rechenweg 1 (Aufzählung): ein Schritt ist ein Besuch der Knotenrekursion für eine Koalition, gezählt über alle 2^d Koalitionen. Rechenweg 2 (Algorithmus 2): ein Schritt ist ein Aufruf der Knotenrekursion "
    "plus je eine Schleifenrunde in EXTEND und UNWIND (Kopien und Summen sind nicht gezählt). Die zwei Einheiten sind nicht gleich teuer: die Zählung zeigt das Wachstum, nicht den Preis."
)
EXP4_CAPTION = (
    "Zähleinheiten: Koalitionen M, Modellauswertungen (M mal 30 Hintergrundzeilen), Instanzen (je Menge 10 Zeilen bei d = 6, 6 bei d = 10), Zufalls-Seeds der Ziehung. Die Spieltabellen sind eingefroren "
    "(`tools/shn_make_games.py`); die Ziehungen und die Regression laufen hier. Fehler = größter |φ̂ − φ| geteilt durch das größte |φ| der Aufzählung, auf demselben Hintergrund: gemessen wird die Näherung, nicht der Abstand zur Wahrheit."
)
EXP5_CAPTION = (
    "Zähleinheiten: Spiele (je 40 Zufallsspiele in fünf Gruppen), Paare (jede gezogene Koalition mit ihrem Komplement). Möbius-Koeffizienten c_T: v(S) = Σ über T ⊆ S von c_T; Ordnung = größte Größe |T| mit c_T ≠ 0."
)

NEGATIV_1 = ("Ehrlicher Negativbefund: liegt x₁ an der Wurzel (81 der 131 Formen bei q = 0,9), bekommt x₂ immer 0; bei q = 0,50 sind x₁ und x₂ unabhängig und alle Funktionale gleich. "
             "Wie oft ein Lernverfahren in die zweite oder dritte Klasse gerät, zeigt Experiment 2.")
NEGATIV_2 = ("Ehrlicher Negativbefund: gierige Merkmalswahl (gieriger Baum, Boosting) bleibt im Toy f = x₁ ohne Rauschen in der interventionalen Klasse, weil schon ein Schnitt auf x₁ die Daten vollständig trennt. Im Vehikel liest das Modell die Art wirklich "
             "(interventional ungleich 0), die Lücke zwischen den Funktionalen ist dort kleiner als im Toy.")
NEGATIV_3 = ("Ehrlicher Negativbefund: Algorithmus 2 lohnt in dieser Zählung erst bei mehr Merkmalen: bei d = 6 und Tiefe 8 braucht er in den acht Bäumen der Tabelle (Erstmessung) mehr Schritte als die Aufzählung "
             "(in der Zweitmessung mit anderen Zufalls-Seeds nur in 6 von 8). Am trainierten Ensemble liegt der pfadabhängige Beitrag im Median nicht bei null, sondern bei dem unten genannten Abstand vom interventionalen.")
NEGATIV_4 = ("Ehrlicher Negativbefund: bei d = 6 kostet die Aufzählung nur 64 Koalitionen; die einfache Stichprobe ist dort mit M = 96 oder mehr Ziehungen teurer als sie, wenn jede Ziehung neu ausgewertet wird (doppelte Ziehungen brauchen nur eine Auswertung, und es gibt nur 62 innere Koalitionen). Der Median verdeckt den Tail: "
             "das 90 %-Quantil steht im Ergebnis, und die Rang-1-Quote kann kleiner als 1 sein.")
NEGATIV_5 = ("Ehrlicher Negativbefund: das Netz hat die Möbius-Ordnung 6, aber der größte Teil seiner Quadratsumme liegt in den Ordnungen 1 und 2, und sein Fehler ist klein, aber nicht null. "
             "Die Reihe der gemessenen Modelle ist kein Gesetz; zu wenige Paare (weniger als d − 1) sind singulär.")

GRENZEN_MARKDOWN = """
| Annahme | Was passiert, wenn sie verletzt ist | Offen |
|---|---|---|
| **Ein erfundenes Vehikel und ein Toy** | Es gibt keine reale Datenquelle. Zahlen gelten für den Verweilzeit-Generator (sechs Merkmale, Zoll und Vorlauf strukturell von der Art abhängig) und das Toy (drei Merkmale). | andere Daten nicht gemessen |
| **scikit-learn-Bäume und -Modelle, eingefroren** | Es wird nicht mit XGBoost, LightGBM oder dem Paket `shap` verglichen; was „Cover“ dort bedeutet und ob Pakete die Aufzählung so ausführen, ist nicht gemessen. Algorithmus 2 ist eine eigene Implementierung, nur über Orakel geprüft. | Paketvergleich |
| **Pfadabhängig nach TS20** | „Baum-SHAP“ heißt hier das pfadabhängige Funktional des Baums nach Lundberg u. a. (Algorithmus 1 und 2). Der interventionale TreeSHAP (O(T·R·L)) ist nicht gebaut und nicht gemessen. | interventionaler TreeSHAP |
| **KernelSHAP: d höchstens 10, Regression, keine Klassifikation** | d = 10 sind sechs Merkmale plus vier unabhängige Rauschspalten (Nullspieler); kein d über 10. Instanzmengen sind klein (10 bzw. 6 Zeilen je Menge, fünf getrennte Mengen). | Klassifikation, größere d |
| **Erklärer-Zufall** | Der Anteil „gleiches Rang 1 wie die Aufzählung“ ist gemessen; ob zwei KernelSHAP-Läufe untereinander dasselbe Rang 1 haben, ist nicht gemessen. | Läufe untereinander |
| **Mindestzahl Paare** | Sie ist nur an Zufallsspielen gemessen, nicht an trainierten Modellen. | trainierte Modelle |
| **Zählungen, keine Zeiten; eine Plattform** | Schritte und Koalitionen sind keine Sekunden. Gemessen unter Windows mit Python 3.14; Linux-Lauf (Docker, Python 3.12) am 2026-11-20 grün. | - |
| **Nutzen von Beitragszahlen** | In den gelesenen Quellen gab es keine Messung, ob Beitragszahlen im Betrieb, in der Logistik oder im Recht zu besseren Entscheidungen führen; hier wird nichts dazu behauptet. | nicht belegt |
"""

IG_ABSATZ = """
**Einordnung: Integrated Gradients.** Auch dieses Verfahren ist eine Näherung mit einem Preis in Schritten: es integriert den Gradienten entlang einer geraden Linie von einer Basis zur Zeile
(Quadratur). Für min(x₁, x₂) mit der Basis 0 und x = (1, 3) gibt es (1, 0), die Baseline-Shapley-Aufteilung aber (1/2, 1/2); für x₁·x₂ geben beide x₁x₂/2 (Handwerte, Quelle:
Sundararajan u. a. 2017, S. 5, im Original gelesen). Die Quadraturregel mit rechten Endpunkten hat einen Fehler von etwa 1/m, die mit Mittelpunkten etwa 1/m² (Handbeispiel f = x₁x₂ + sin x₁, Fehler der Vollständigkeit
0,357 bei m = 5 Stützstellen gegen 0,00375 bei Mittelpunkten; 0,00874 bei m = 200 gegen 0,0000023). Wie stark die Basis den Beitrag verschiebt, ist an **einem** Netz gemessen und nicht allgemein belegt.
Alles Weitere zu Integrated Gradients und Torch-Modellen gehört nicht in diese Demo.
"""

MATHE_LESEHILFE = r"""
**Lesehilfe.** *Gegeben:* eine Zeile x mit d Merkmalen, ein Modell f (oder ein Baum mit Blattwerten und Trainingsanteilen) und ein Wertfunktional, das jeder Koalition S ⊆ {1, …, d} einen Wert v(S) zuordnet.
*Gesucht:* der Shapley-Wert φ_i jedes Merkmals, entweder aus allen 2^d Werten (Aufzählung) oder aus wenigen gezogenen (KernelSHAP).

| Symbol | Bedeutung |
|---|---|
| d | Zahl der Merkmale (Spieler) |
| S | Koalition: Teilmenge der Merkmale, deren Werte als bekannt gelten |
| v(S) | Wert der Koalition (hängt vom Wertfunktional ab) |
| φ_i | Shapley-Wert (Beitrag) des Merkmals i |
| x_S, b | Werte der Zeile in S bzw. Zeile des Hintergrunds |
| c_T | Möbius-Koeffizient der Menge T; Ordnung des Spiels = größte Größe \|T\| mit c_T ≠ 0 |
| M | Zahl der von KernelSHAP gezogenen Koalitionen |
| s | Größe einer Koalition, \|S\| |
"""

MATHE_BLOECKE = [
    (r"""**Shapley-Wert (Koalitionsformel).** Für jedes Merkmal i gilt""",
     r"""\varphi_i \;=\; \sum_{S \subseteq N \setminus \{i\}} \frac{|S|!\,(d-|S|-1)!}{d!}\,\bigl(v(S \cup \{i\}) - v(S)\bigr)"""),
    (r"""**Interventional** setzt den Hintergrund ein, **konditional** bedingt, **pfadabhängig** (Baum, Algorithmus 1 von Lundberg u. a.) folgt der Zeile an den Knoten, deren Merkmal in S liegt, und mischt sonst die beiden Kinder
nach ihrer Trainingsmasse $m$:""",
     r"""v_{\text{pfad}}(S) \;=\; \begin{cases} \text{Blattwert} & \text{am Blatt} \\ v(\text{Kind, dem } x \text{ folgt}) & \text{Merkmal des Knotens} \in S \\ \dfrac{m_{\text{links}}\, v(\text{links}) + m_{\text{rechts}}\, v(\text{rechts})}{m_{\text{links}} + m_{\text{rechts}}} & \text{sonst} \end{cases}"""),
    (r"""**Der Shapley-Kern (KernelSHAP).** Die Größe $s$ einer gezogenen Koalition hat die Wahrscheinlichkeit proportional zu""",
     r"""\pi(s) \;=\; \frac{d-1}{\binom{d}{s}\, s\,(d-s)}, \qquad s = 1, \dots, d-1"""),
    (r"""Gelöst wird die Regression mit der Nebenbedingung $\sum_i \varphi_i = v(N) - v(\emptyset)$ (die letzte Unbekannte wird eliminiert):""",
     r"""\min_{\varphi}\; \sum_{S \text{ gezogen}} \Bigl(v(S) - v(\emptyset) - \sum_{i \in S} \varphi_i\Bigr)^2"""),
    (r"""**Möbius-Koeffizienten.** Jedes Spiel lässt sich eindeutig schreiben als""",
     r"""v(S) \;=\; \sum_{T \subseteq S} c_T, \qquad \varphi_i \;=\; \sum_{T \ni i} \frac{c_T}{|T|}"""),
    (r"""**Warum gepaartes Ziehen bis Ordnung 2 exakt ist (eigene Rechnung, nicht aus einer Quelle gelesen).** Mit $s_i = 2z_i - 1 \in \{-1, +1\}$ für die Koalition $z$ gilt""",
     r"""z_i z_j \;=\; \frac{s_i s_j + s_i + s_j + 1}{4}"""),
    (r"""Der **ungerade Teil** eines Spiels der Ordnung höchstens 2 ist linear in $s$ mit genau den Shapley-Werten als Koeffizienten:""",
     r"""o(z) \;=\; \frac{v(z) - v(1-z)}{2} \;=\; \frac{1}{2}\,\varphi \cdot s"""),
]
MATHE_SCHLUSS = (
    "Die Regression des gepaarten Ziehens (gleiches Gewicht für z und 1 − z) zerlegt sich in den geraden Teil, den die Nebenbedingung festlegt, und den ungeraden Teil o(z) − ½ s·φ̂, der bei φ̂ = φ null ist; "
    "sind die gezogenen Vektoren s unabhängig genug (dafür braucht man mindestens d − 1 verschiedene Paare), ist die Lösung eindeutig, also φ̂ = φ, gleich welche Koalitionen gezogen wurden. "
    "Bei Ordnung 3 enthält o(z) einen Term s_i s_j s_k, der auf den gezogenen Punkten mit linearen s verwechselt werden kann: für d = 4 und Paare der Größen 1 und 3 gilt s₀s₁s₂ = −s₃, daher ist dort für das Spiel mit c₀₁₂ = 1 (v(S) = 1, wenn {0, 1, 2} ⊆ S) φ̂ = (3/8, 3/8, 3/8, −1/8) statt (1/3, 1/3, 1/3, 0). "
    "**Skizze, kein vollständiges Modell:** Algorithmus 2 (EXTEND und UNWIND) ist hier nicht ausgeschrieben; seine Rechnung steht in `shn_treeshap.py` nach Lundberg u. a. (Algorithmus 2) und ist gegen die Aufzählung, "
    "alle Permutationen und Brüche geprüft. Die Schranke „Ordnung höchstens Baumtiefe“ ist an Zufallsbäumen in Brüchen und an trainierten Modellen **gemessen**, kein Satz aus einer Quelle."
)

LITERATUR = """
**Literatur** (Stufe: gelesen = Original im Wortlaut gelesen). Lundberg u. a.: *Explainable AI for Trees: From Local Explanations to Global Understanding* (arXiv, Fassung 1): Algorithmus 1 und 2, gelesen.
Lundberg, Lee 2017: *A Unified Approach to Interpreting Model Predictions*: Satz 2 (Shapley-Kern), gelesen. Janzing u. a. 2019 (arXiv v2): Beispiel 1 (irrelevantes, abhängiges Merkmal bei bedingter Wertfunktion), gelesen.
Kumar u. a. 2020 (Tabelle 1: KernelSHAP als interventional, TreeSHAP als pfadabhängig eingeordnet), gelesen. Sundararajan u. a. 2017 (Integrated Gradients, S. 5), gelesen.
Nur als Beschreibung genutzt, nicht im Wortlaut gelesen: Molnar, *Interpretable Machine Learning*, Kap. 17 und 18; Covert, Lee 2021 (nur Zusammenfassung; ob dort steht, dass gepaartes Ziehen für niedrige Ordnung exakt ist: nicht gelesen);
Hwang u. a. 2026 (unbegutachtet). Vorbilder für nachrechenbare Erklärer-Vergleiche: XAI-Bench, OpenXAI, Hooker u. a.; **neu ist hier keine Aussage, nur die nachrechenbare Vorführung mit Zähleinheiten und Orakeln.**
"""
