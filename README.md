# Wie wird der Beitrag ausgerechnet? Baum-SHAP und KernelSHAP – Streamlit-Demo

<!-- TODO nach dem Deploy: direkt hierher den Live-Link setzen (Leerzeile davor und danach):
**[→ Demo live ausprobieren](https://sebastianhanisch-shap-naeherung-demo.streamlit.app/)** -->

Diese Demo gehört zur Reihe „Beitrag und Beleg“ (Modelle verstehen) in der „Konzepte“-Reihe für die Website „Sebastian Hanisch – Operations Research und Machine Learning“. Sie ist ohne die übrigen Demos der Reihe lesbar.
Sie zeigt, dass der **Rechenweg** für Beitragszahlen (Shapley-Werte; „SHAP“ heißt die Software und ihre Näherungen) **seine eigene Struktur mitbringt**: Baum-SHAP rechnet das pfadabhängige Funktional des **Baums**, nicht das der Funktion, und gepaartes KernelSHAP ist bis zur Ordnung 2 der Wechselwirkungen exakt (bei nicht singulärer Ziehung); wie groß der Fehler darüber ist, hängt hier am Anteil der höheren Ordnungen.
Das Vehikel ist **erfunden**: ein Verweilzeit-Generator für Container (sechs Merkmale) und ein binäres Handbeispiel (Toy, drei Merkmale). **Es gibt keine reale Datenquelle.**

## Kernfrage

(1) Rechnet Baum-SHAP den Beitrag der Funktion oder den des Baums? (2) Was kostet es, den Beitrag auszurechnen? (3) Wann ist die Näherung exakt, wann nicht? Gezählt werden Baumformen, Zufalls-Seeds, Instanzen, Koalitionen, Modellauswertungen und Schritte, **keine Sekunden**.
Die Aussagen gelten für dieses erfundene Vehikel und dieses Toy, für scikit-learn-Bäume ohne Gewichtung, für die Zufalls-Seeds, Instanzmengen und Baumformen, die bei jeder Zahl stehen; **was ein bestimmtes Paket (`shap`, XGBoost, LightGBM) rechnet, ist nicht gemessen.** Neu ist keine Aussage, nur die nachrechenbare Vorführung mit Zähleinheiten und Orakeln (Vorbilder: XAI-Bench, OpenXAI, Hooker u. a.).

## Befunde und Korrekturen gegenüber Plan und Vormessung

Die Zahlen der Vormessung sind Erwartungen; im Bau wurden sie mit dem Zufallsgenerator der Demo (SplitMix64 für die Ziehungen) und einer **Zweitmessung mit anderen Zufalls-Seeds** (`data/*_zweit.json`) neu gemessen. Was sich verschoben hat:

* **Die Modellwerte sind nicht überall exakt f.** Bei den Wäldern mit einem und mit zwei Merkmalen je Schnitt (n = 400, ohne Rauschen) ist die größte Abweichung von f über 30 Zufalls-Seeds und acht Punkte 0,0167 (ein Sechzigstel: genau ein Baum des Waldes gibt dort einen anderen Wert); der pfadabhängige Beitrag für x₂ ist trotzdem 0,099. Bei n = 4 000 und bei gierigem Baum und Zufallsbaum ist sie 0. Die Vormessung nannte die Modellwerte „exakt f“. Boosting weicht um höchstens 0,00002 ab (100 Bäume, Lernrate 0,1: der Rest nach 100 Schritten).
* **„Algorithmus 2 braucht bei d = 6 und Tiefe 8 mehr“ gilt in der Erstmessung in 8 von 8 Bäumen, in der Zweitmessung in 6 von 8.** Das Verhältnis (Aufzählung geteilt durch Algorithmus 2) liegt in der Zweitmessung bei 0,55 bis 1,05.
* **„Pfadabhängig und interventional haben nie ein anderes Vorzeichen des größten Beitrags“** gilt in der Erstmessung (0 von je 150 Zeilen in allen fünf Mengen und drei Modellen), in der Zweitmessung bei beiden Wäldern aber nicht: in einer Menge 1 von 150 Zeilen (0,7 %).
* **Gieriger Baum mit Rauschen 0,5, n = 4 000:** der Anteil der Zufalls-Seeds mit |φ₂| > 0,02 ist in der Erstmessung 0 von 30, in der Zweitmessung 1 von 30.
* **Die Ziehungen von KernelSHAP gelten für alle Zeilen einer Instanzmenge gemeinsam** (die Vormessung zog je Zeile neu); das macht die Läufe in der Menge abhängig, die Mediane aber nicht anders: gepaart, d = 6, 24 Koalitionen, Boosting Tiefe 3 jetzt 0,005 bis 0,019 (Vormessung 0,005 bis 0,020), Tiefe 4 0,013 bis 0,043 (0,015 bis 0,048), Wald 0,037 bis 0,076 (0,038 bis 0,078), Netz 0,018 bis 0,042 (0,021 bis 0,042); d = 10, 40 Koalitionen: Boosting 0,011 bis 0,021 (0,011 bis 0,018), Wald 0,039 bis 0,076 (0,032 bis 0,099), Netz 0,036 bis 0,053 (0,031 bis 0,060).
* **Neu gezählt: singuläre Ziehungen.** Gepaart mit 40 Koalitionen bei d = 10 war in jedem der drei Modelle 1 von 30 Ziehungen singulär (die gezogenen Koalitionen spannen den Raum nicht auf, die Lösung ist nicht eindeutig); bei d = 6 und 24 Koalitionen gab es keine. Die Vormessung hat das nicht gezählt.
* **Zufallsspiele:** für Ordnung 1 und 2 ist das gepaarte Ziehen in jeder Zelle **genau dann exakt, wenn die Ziehung nicht singulär ist** (Zahl exakter plus Zahl singulärer Spiele gleich 200; Vormessung: nicht untersucht). Die Gruppenbänder sind breit (d = 6, Ordnung 2, 8 Paare: 0,82 bis 0,88, in der Zweitmessung 0,82 bis 0,93; Vormessung 0,80 bis 0,85; bei etwa 0,85 und 40 Spielen je Gruppe ist der Standardfehler etwa 0,06).
* **Möbius-Ordnung:** jetzt fünf Instanzmengen zu 10 Zeilen (Vormessung drei Zufalls-Seeds); die Reihe Ordnung gleich Baumtiefe 1, 2, 3, 4 und Netz und unbegrenzter Wald mit Ordnung 6 bleibt. Wald Tiefe 2 hat ein R² gegen μ von nur 0,443 (schwaches Modell, trotzdem Ordnung 2), Boosting Tiefe 1 0,704.
* **Integrated Gradients** ist ein Einordnungsabsatz (Handwerte und Quadraturregel, `tests/test_integrated_gradients.py`), kein Experiment.

## Modell

**Vehikel (erfunden).** Container haben Art (Import, Export, Umschlag, Leer), Kunde (12), Zollhalt, Wochenende, Vorlauf und ein reines Rauschmerkmal; die mittlere Log-Verweilzeit μ(x) folgt einer bekannten Formel. Zollhalt gibt es nur bei Import, Vorlauf nur bei Export und Umschlag: das erzeugt die Abhängigkeit, an der sich der Rechenweg des Baum-SHAP verrät. Der Generator ist aus `verweilzeit-demo` **kopiert** (`shn_generator.py`, Kopie-Test gegen eingefrorene Werte). **Toy (Handbeispiel-Welt, erfunden).** Drei binäre Merkmale, x₁ gleichverteilt, x₂ gleich x₁ mit Wahrscheinlichkeit 9/10, x₃ unabhängig; f = x₁; acht Punkte, vollständig aufgezählt.
**Konventionen.** Ein **Spiel** je Zeile, Spieler sind die Merkmale; **Wertfunktionale** interventional, konditional, pfadabhängig (Leave-one-out kommt hier nicht vor). **Beitrag** (je Zeile) gegen **Wichtigkeit** (Mittel von |φ|, gewichtet mit der Verteilung). „Wahrheit“ heißt nur „Wahrheit je Funktional der bekannten Funktion“. **Gleichstandsregel der Bäume:** scikit-learn wählt bei gleichem Gewinn nach der Reihenfolge der Merkmale und seinem Zufalls-Seed; die Zahlen hängen daran, darum stehen Spannen über Zufalls-Seeds. **Fehlermaß KernelSHAP:** größter |φ̂ − φ| je Merkmal geteilt durch das größte |φ| der Aufzählung auf demselben Hintergrund (30 Trainingszeilen); der Fehler misst die Näherung des Rechenwegs, nicht den Abstand zur Wahrheit. Rangmaß: Anteil der Läufe mit gleichem Rang-1-Merkmal wie die Aufzählung (das Betragsmaß ist der Fehler).

## Methodik

* **Rechenweg 1, Aufzählung (Lundberg u. a., Algorithmus 1):** v(S) für alle 2^d Koalitionen; **Schritt** = ein Besuch der Knotenrekursion für eine Koalition. **Rechenweg 2 (Algorithmus 2):** ein Durchgang mit Pfadgewichten; **Schritt** = ein Aufruf der Knotenrekursion plus je eine Schleifenrunde in EXTEND und UNWIND (Kopien und Summen sind nicht gezählt). Von Hand: ein Baum mit einem Merkmal und zwei Blättern gibt 5 Schritte (Aufzählung) und 7 (Algorithmus 2), der vollständige Baum mit d = 2 gibt 19 und 33. Die zwei Einheiten sind nicht gleich teuer: die Zählung zeigt das Wachstum, nicht den Preis.
* **KernelSHAP** (eigene Implementierung, kein Paket): Größe der Koalition nach dem Shapley-Kern ziehen, Teilmenge zufällig, ungewichtete kleinste Quadrate mit der Nebenbedingung Σφ = v(N) − v(∅). **Einfach:** M Ziehungen mit Zurücklegen. **Gepaart:** M/2 Ziehungen und ihre Komplemente. Mit **allen** Koalitionen und Shapley-Gewicht ist es exakt (Orakel).
* **Möbius-Koeffizienten** c_T mit v(S) = Σ über T ⊆ S von c_T; **Ordnung** des Spiels = größte Größe |T| mit c_T ≠ 0; φ_i = Σ über T ∋ i von c_T / |T| (zweite Definition desselben Werts, als Orakel). Warum gepaartes Ziehen bis Ordnung 2 exakt ist: **eigene Rechnung**, nicht aus einer Quelle gelesen (ungerader Teil linear in s = 2z − 1, im 📐-Expander der App).
* **Frozen:** scikit-learn-Modelle und Spieltabellen sind eingefroren (`data/`, Quelle und Stand je Datei im Feld `meta`; `tools/shn_make_learn.py`, `shn_make_costs.py`, `shn_make_games.py`, je mit `--check`, `--zweit`). Live in der App: die 131 Baumformen, beide Rechenwege an einem eingefrorenen Baum, die Ziehungen und die Regression von KernelSHAP und die Zufallsspiele.

## Befunde (gemessen, keine Behauptungen)

Jede Zahl steht mit ihrer Zähleinheit und der Zeilenmenge; Aussagen über Beiträge je Zeile nennen die Spanne über **fünf getrennte Instanzmengen** (Zeilenmengen: natürliche Zeilen des erfundenen Generators). Die Kennung in eckigen Klammern verweist auf die Tabelle „Zahl → Test“ weiter unten.

**1. Eine Funktion, viele Bäume** (Zähleinheit: Baumformen, Toy, q = 0,9). [B1] Alle 131 Binärbäume, die f = x₁ auf allen acht Punkten berechnen (1, 10 und 120 nach Tiefe 1, 2, 3): pfadabhängig geben **97** dem nie gelesenen x₂ den Beitrag 0, **26** den konditionalen Wert 0,200 (1/5) und **8** den Wert 0,100 (1/10); der Beitrag ist ein Fünftel der Masse der Blätter, auf deren Pfad x₂ vor x₁ getrennt wird (131 von 131 Bäumen, exakt in Brüchen). Wurzel x₁: 81 Bäume, alle mit Beitrag 0; Wurzel x₂: 25 Bäume, alle 0,200. Wahrheit der Funktion: interventional (0,5; 0; 0), konditional (0,34; 0,2; 0). Das Mini-Beispiel mit zwei Merkmalen von Hand: Baum A (nur x₁) (1/2, 0), Baum B (x₂ zuerst) (3/10, 1/5).

**2. Was macht ein Lernverfahren daraus?** (Zähleinheit: Zufalls-Seeds = Trainingsdatensätze, 30 im Toy, 5 im Vehikel.) [B2] Toy, n = 400, ohne Rauschen: gieriger Baum (Tiefe 3) und Boosting (100 Bäume, Tiefe 3) pfadabhängig und interventional (0,500; 0,000; 0,000), 0 von 30 Zufalls-Seeds mit |φ₂| > 0,02; Zufallsbaum (ein zufälliges Merkmal, zufälliger Schnitt) pfadabhängig (0,420; 0,100; 0,009), 18 von 30; **Wald (60 Bäume) mit einem Merkmal je Schnitt (0,421; 0,099; 0,003)**, mit zwei Merkmalen (0,447; 0,066; 0,001), je 30 von 30; interventional überall (0,500; 0,000; 0,000). n = 4 000 gibt dieselben Zahlen auf zwei Stellen. [B3] Mit Rauschen 0,5: gieriger Baum n = 400 pfadabhängig (0,496; 0,031; 0,026) gegen interventional (0,496; 0,043; 0,027), 22 von 30; n = 4 000 (0,500; 0,008; 0,008) gegen (0,501; 0,011; 0,008), 0 von 30; Wald mit einem Merkmal n = 4 000 (0,422; 0,097; 0,009) gegen (0,502; 0,012; 0,008), 30 von 30. [B4] Vehikel, Ziel g = Vorlauf (liest nur den Vorlauf; Art und Zoll hängen strukturell an ihm), 5 Zufalls-Seeds, fünf Instanzmengen zu je 25 Zeilen: gieriger Baum Tiefe 6 (R² gegen g 0,9998), Wald mit allen Merkmalen je Schnitt (1,0000) und Boosting (1,0000) geben der Art höchstens 0,0012; **Wald mit der Hälfte der Merkmale je Schnitt (R² 0,9980): Art interventional 0,540 bis 0,639, pfadabhängig 0,711 bis 0,813; mit einem Merkmal (0,9940): 0,672 bis 0,797 und 0,859 bis 0,983**; dort liest das Modell die Art wirklich, die Lücke zwischen den Funktionalen ist 0,13 bis 0,21, kleiner als im Toy. Wahrheit von g: interventional 0 für alles außer dem Vorlauf, konditional Art 0,94 bis 1,18.

**3. Baum-SHAP, zwei Rechenwege, Kosten und Abstand** (Zähleinheit: Schritte je Instanz nach der Zählregel oben; acht Bäume je Zelle, vier bei d = 14; erfundene Zielfunktion über d gleichverteilten Merkmalen, n = 3 000). [B5] Algorithmus 2 und Aufzählung geben dieselben Beiträge: größte Abweichung 1,8·10⁻¹⁵ in den 15 Zellen der Tabelle, in 131 × 8 Fällen in Brüchen exakt; auch bei Merkmalen, die im Pfad mehrfach vorkommen. [B6] **d = 14, Tiefe 3: Aufzählung 133 120 Schritte, Algorithmus 2 96 bis 121 (vier Bäume). d = 6, Tiefe 8: Algorithmus 2 braucht in 8 von 8 Bäumen der Erstmessung (6 von 8 in der Zweitmessung) mehr (6 084 bis 9 509 gegen 5 074 bis 5 828)**; d = 6, Tiefe 6 in 1 von 8; d = 8, Tiefe 8 weniger (Verhältnis 1,76 bis 3,61). Vollständiger Baum der Tiefe d über d Merkmalen: Aufzählung 3^(d+1) − 2^(d+1), Algorithmus 2 Σ_{k=0..d} 2^k (1 + k) + 2^d d² (d = 10: 175 099 und 122 881). [B7] Am trainierten Ensemble (n = 1 000, 5 Zufalls-Seeds, fünf Instanzmengen zu je 30 Zeilen, Hintergrund = exakte Bevölkerung des Generators): relativer Abstand pfadabhängig gegen interventional im Median Boosting (R² gegen μ 0,927) 6,5 bis 7,6 %, Wald mit der Hälfte der Merkmale (0,894) 15,0 bis 19,9 %, Wald mit allen Merkmalen (0,894) 13,0 bis 15,6 %; anderes Rang 1 in 1,3 bis 8,7 %, 3,3 bis 14,0 % und 5,3 bis 10,0 % der Zeilen; anderes Vorzeichen des größten Beitrags: in der Erstmessung 0 von 150 Zeilen je Menge, in der Zweitmessung bei beiden Wäldern in einer Menge 1 von 150 Zeilen.

**4. KernelSHAP: Kosten gegen Fehler** (Zähleinheiten: Koalitionen, Modellauswertungen = Koalitionen mal 30 Hintergrundzeilen, Ziehungen; Medianfehler je Instanzmenge, Spanne über fünf Mengen; d = 6: 10 Zeilen und 10 Ziehungen je Menge, d = 10: 6 Zeilen und 6 Ziehungen). [B8] **d = 6, 24 Koalitionen = 720 von 1 920 Auswertungen (37,5 %):** Boosting Tiefe 1 und 2 gepaart 0,000 (einfach Tiefe 2: 0,038 bis 0,085); Tiefe 3 gepaart 0,005 bis 0,019 (einfach 0,083 bis 0,168); Tiefe 4 gepaart 0,013 bis 0,043 (einfach 0,089 bis 0,146); Wald Tiefe 2 gepaart 0,000; unbegrenzter Wald 0,037 bis 0,076 (einfach 0,127 bis 0,223); Netz 0,018 bis 0,042 (einfach 0,091 bis 0,143). [B9] **d = 10, 40 Koalitionen = 1 200 von 30 720 Auswertungen (3,9 %), gepaart:** Boosting Tiefe 3 0,011 bis 0,021, Wald 0,039 bis 0,076, Netz 0,036 bis 0,053; einfach 0,068 bis 0,155, 0,104 bis 0,279 und 0,119 bis 0,223; die einfache Stichprobe braucht 1 280 Ziehungen (125 % der 1 024 Koalitionen der Aufzählung, teurer als sie, wenn jede Ziehung neu ausgewertet wird; doppelte Ziehungen brauchen nur eine Auswertung, die Zahl verschiedener Koalitionen ist nicht gezählt) für 0,013 bis 0,023, 0,022 bis 0,040 und 0,019 bis 0,037. Der Median verdeckt den Tail: das 90 %-Quantil des gepaarten Fehlers beim unbegrenzten Wald (d = 6, 24 Koalitionen) ist 0,182. [B10] Orakel: alle Koalitionen mit Shapley-Gewicht gleich Aufzählung bis 10⁻⁹ in jedem eingefrorenen Modell; mit 2 Koalitionen (ein Paar) ist jede Ziehung singulär.

**5. Die Ordnung entscheidet** (Zähleinheit: Spiele; je Zelle 5 Gruppen zu 40 Zufallsspielen). [B11] **Möbius-Ordnung des interventionalen Spiels (d = 6, fünf Mengen zu 10 Zeilen):** Boosting Tiefe 1, 2, 3, 4 haben Ordnung 1, 2, 3, 4 (Anteil der Quadratsumme in Ordnung 1 und 2: 1,000 / 0,945 und 0,055 / 0,851 und 0,144 / 0,811 und 0,166), Wald Tiefe 2 Ordnung 2, unbegrenzter Wald und Netz Ordnung 6 (Anteil der Ordnungen ab 3 an der Quadratsumme: 6,1 % und 1,9 %); in Brüchen ist die Ordnung bei 180 Zufallsbäumen höchstens die Tiefe, und sie erreicht die Tiefe. Das ist eine Reihe von sieben Modellen, **kein Gesetz**. Die Ordnung entscheidet, ob gepaartes KernelSHAP exakt ist; wie groß der Fehler jenseits von Ordnung 2 ist, hängt zusätzlich am Anteil der Quadratsumme in den höheren Ordnungen: das Netz hat Ordnung 6 und einen Medianfehler (24 Koalitionen, d = 6), der dem von Boosting Tiefe 4 (Ordnung 4) ähnelt und kleiner ist als der des unbegrenzten Waldes (Ordnung 6), siehe [B8]. [B12] **Gepaartes Ziehen, Zufallsspiele:** Ordnung 3 ist in **keinem** der 200 Spiele je Zelle exakt (d = 6 mit 4 bis 16, d = 10 mit 8 bis 20 Paaren); Ordnung 2 ist bei d = 6 mit 5 / 8 / 12 / 16 Paaren in 61 / 170 / 196 / 199 von 200 Spielen exakt (Zweitmessung 67 / 171 / 198 / 200), bei d = 10 mit 9 / 12 / 16 / 20 Paaren in 36 / 147 / 192 / 198 (Zweitmessung 32 / 143 / 191 / 198); mit weniger als d − 1 Paaren (4 bei d = 6) ist jede Ziehung singulär. Handwerte (Spiel mit c₀₁₂ = 1, also v(S) = 1, wenn {0, 1, 2} ⊆ S, sonst 0): Ordnung 3 bei d = 4 mit Paaren der Größen 1 und 3 gibt (3/8, 3/8, 3/8, −1/8) statt (1/3, 1/3, 1/3, 0), alle 14 Koalitionen ungewichtet (5/16, 5/16, 5/16, 1/16).

**Einordnung: Integrated Gradients** (kein Experiment, keine App-Rechnung; Handwerte und Quadratur in `tests/test_integrated_gradients.py`). [B13] Für min(x₁, x₂) mit Basis 0 und x = (1, 3) gibt Integrated Gradients (1, 0), die Baseline-Shapley-Aufteilung (1/2, 1/2); für x₁·x₂ beide x₁x₂/2. Quadratur am Handbeispiel f = x₁x₂ + sin x₁ (x = (1,5; −0,7)): der Fehler der Vollständigkeit ist mit rechten Endpunkten 0,357 bei m = 5 und 0,00874 bei m = 200, mit Mittelpunkten 0,00375 und 0,0000023 (etwa 1/m gegen etwa 1/m²). Wie stark die Basis den Beitrag verschiebt, ist an **einem** Netz gemessen (Vormessung) und nicht allgemein belegt.

## Zahl → Test

| Kennung | Zahl (Kurzform) | Test (Band oder exakt) |
|---|---|---|
| B1 | 131 Baumformen, 97 / 26 / 8, Regel Masse/5, Wurzeln 81 und 25, Wahrheit, Mini-Beispiel | `test_handbeispiele.py` (exakt, Brüche), `test_claims.py::test_b1_*` (exakt) |
| B2 | Toy ohne Rauschen: 0,500 / 0,000 / 0,000, Zufallsbaum, Wald; Anteile 0, 18, 30 von 30 | `test_claims.py::test_b2_*` (Band 0,0006 auf die eingefrorenen Zahlen) |
| B3 | Toy mit Rauschen 0,5 | `test_claims.py::test_b3_*` (Band) |
| B4 | Vehikel Ziel Vorlauf: R², Art interventional und pfadabhängig, Lücke, Wahrheit | `test_claims.py::test_b4_*` (Band), `test_make_tools.py` (Neu-Rechnen, Wahrheit gegen Stichprobe) |
| B5 | Algorithmus 2 gleich Aufzählung (1,8·10⁻¹⁵, Brüche, Wiederholung im Pfad) | `test_handbeispiele.py`, `test_oracle_baum_shap.py` (exakt bzw. 1e-10), `test_claims.py::test_b5_*` |
| B6 | Kostentabelle, geschlossene Formen, 5 / 7 und 19 / 33, 175 099 und 122 881 | `test_handbeispiele.py` (exakt), `test_claims.py::test_b6_*` (exakt auf die eingefrorenen Zähler), `test_data.py` (Zähler neu gezählt) |
| B7 | Abstand am Ensemble: Median, Rang 1, Vorzeichen | `test_claims.py::test_b7_*` (Band 0,0006) |
| B8 | KernelSHAP d = 6, 24 Koalitionen | `test_claims.py::test_b8_*` (Band 0,0006, Zufall aus SplitMix64) |
| B9 | KernelSHAP d = 10, 40 und 1 280 Koalitionen, 90 %-Quantil | `test_claims.py::test_b9_*` (Band 0,0006) |
| B10 | Orakel alle Koalitionen, ein Paar singulär | `test_oracle_kernelshap.py`, `test_edge_cases.py` |
| B11 | Möbius-Ordnungen und Anteile, Brüche bei 180 Zufallsbäumen | `test_claims.py::test_b11_*` (Band 0,0006), `test_handbeispiele.py` (Brüche) |
| B12 | Zufallsspiele: 0 von 200, 61 / 170 / 196 / 199, 36 / 147 / 192 / 198 (Zweitmessung 67 / 171 / 198 / 200, 32 / 143 / 191 / 198); Handwerte | `test_claims.py::test_b12_*` (Zählung exakt, Toleranz 2 wegen der Plattform), `test_handbeispiele.py` (Brüche) |
| B13 | Integrated Gradients: Handwerte, Quadratur 0,357 / 0,00375 / 0,00874 / 0,0000023 | `test_integrated_gradients.py` (Band 2 %, Handwerte exakt) |
| Z | Zweitmessung (Befunde und Korrekturen) | `test_claims.py::test_zweitmessung_*` |
| P | Hilfetexte der Presets (0,1; 1/5; 3,9 %; 175 099) | `tests/test_presets.py` |

Messungen dieser README, die **nicht** im Test stehen: die Vergleichswerte der Vormessung in Klammern unter „Befunde und Korrekturen“ (Zahlen der Planung, nicht dieser Demo) und die Laufzeiten (Sekunden sind kein Befund).

## Ehrliche Grenzen

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

## Bewusst nicht enthalten / nicht gemessen

* 🚧 **Pakete:** `shap`, XGBoost und LightGBM sind weder installiert noch verglichen; jede Aussage über ein Paket fehlt mit Absicht. 🚧 **Interventionaler TreeSHAP** (Lundberg u. a., interventionale Variante im Methodenteil): ein Einordnungssatz, nicht gebaut, nicht gemessen.
* 🚧 **Erklärer-Zufall von KernelSHAP:** ob zwei Läufe untereinander dasselbe Rang 1 haben, ist nicht gemessen (nur der Anteil gleich der Aufzählung). 🚧 **Mindestzahl Paare an trainierten Modellen** und **Klassifikation** nicht gemessen; **d über 10** nicht gemessen.
* 🚧 **Lernverfahren:** nur gieriger Baum, Zufallsbaum, Wald (ein und zwei Merkmale je Schnitt im Toy, die Hälfte oder alle im Vehikel) und Boosting; keine XGBoost-Verfahren; das Toy hat nur ein abhängiges Paar. 🚧 **Plattform:** gemessen unter Windows und Python 3.14; Linux-Lauf (Docker, Python 3.12) am 2026-11-20 grün. 🚧 **Wirkung in Betrieb:** nicht belegt.
* **Abgrenzung zu bestehenden Demos:** `shapley-demo` (Axiome, Aufzählung gegen Reihenfolgen, Stichprobe über Reihenfolgen) hat die Axiome und die Reihenfolgen-Stichprobe; hier steht die Regression über Koalitionen und ihre Kosten. `random-forest-demo` und `cart-demo` (mtry, gieriger Baum, Schnitte auf Rauschmerkmale) trainieren, um zu lernen; hier beeinflusst die Merkmalswahl die Baumform und damit den pfadabhängigen Beitrag.

## Tests

Die Suite besteht aus Handbeispielen in Brüchen (`test_handbeispiele.py`), Orakeln mit anderem Rechenweg (`test_oracle_baum_shap.py`, `test_oracle_kernelshap.py`), Eigenschaften und Randfällen (`test_kernel.py`, `test_edge_cases.py`), den Daten (`test_data.py`, `test_generator_kopie.py`, `test_make_tools.py`), den Claims dieser README (`test_claims.py`), Verdikten, Presets, Layout, Footer, App (AppTest), Laufzeit-Importen und Texten. Die Suite hat 157 Tests; sie lief lokal mit `-n 4` (Windows, Python 3.14, Maschine unter Last durch parallele Arbeiten) in rund 55 Sekunden, ohne `-n` deutlich länger (die AppTest-Läufe überwiegen). Die Neu-Rechnen-Tests (`test_make_tools.py`) vergleichen als Band.

## Dateistruktur

| Datei | Zweck |
|---|---|
| `.github/workflows/tests.yml` | CI: Push, Pull Request, wöchentlicher Canary |
| `.gitignore` | venv, Caches, Streamlit-Secrets |
| `.streamlit/config.toml` | Streamlit-Einstellungen (keine Nutzungsstatistik) |
| `app.py` | Streamlit-App: Kernfragen, fünf Experimente, Grenzen, Formeln |
| `requirements.txt`, `requirements-dev.txt` | Laufzeit (streamlit, numpy, plotly) bzw. plus pytest, scikit-learn, scipy für Tests und Werkzeuge |
| `data/shn_learn.json`, `shn_costs.json`, `shn_games_d6.json`, `shn_games_d10.json` (und `*_zweit.json`) | eingefrorene Messungen mit Quelle und Stand |
| `tools/shn_make_learn.py`, `shn_make_costs.py`, `shn_make_games.py`, `shn_make_common.py` | Erzeugen und Prüfen (`--check`, `--zweit`) der eingefrorenen Dateien (Dev-venv) |
| `shn_tree.py` | Baum als Felder, Aufzählung (Algorithmus 1) mit Besuchszähler, Blattrechnung als zweiter Weg |
| `shn_treeshap.py` | Algorithmus 2 mit Schrittzähler (float oder Brüche), geschlossene Formen |
| `shn_kernel.py` | KernelSHAP: Ziehen (einfach, gepaart), Regression, Singularität, Orakel mit allen Koalitionen |
| `shn_moebius.py` | Möbius-Transformation, Ordnung, Zufallsspiele |
| `shn_shapley.py` | Koalitionsformel (Brüche oder float, schnelle numpy-Fassung) |
| `shn_toy.py` | Toy: Verteilung, die 131 Baumformen, Beiträge, Wahrheit je Funktional |
| `shn_generator.py` | Kopie des Verweilzeit-Generators aus `verweilzeit-demo` |
| `shn_rng.py` | SplitMix64 (Vorlage aus dem Website-Repo, auf jeder Plattform bitgleich) |
| `shn_data.py`, `shn_experiments.py` | Laden der Daten, Rechnungen der fünf Experimente |
| `shn_evaluation.py` | Verdikte als reine Funktionen |
| `shn_constants.py`, `shn_presets.py`, `shn_layout.py`, `shn_visualization.py`, `shn_texts.py` | Regler, Presets mit Permalink, Kopf, Abbildungen, Texte |
| `tests/` | siehe Abschnitt Tests |

## Quellen

Stufe W = im Wortlaut gelesen. **W:** Lundberg u. a., *Explainable AI for Trees: From Local Explanations to Global Understanding* (arXiv, Fassung 1), Algorithmus 1 und 2 und die interventionale Variante im Methodenteil; Lundberg, Lee 2017, *A Unified Approach to Interpreting Model Predictions*, Satz 2; Janzing u. a. 2019 (arXiv v2), Beispiel 1; Kumar u. a. 2020, Tabelle 1; Sundararajan u. a. 2017, S. 5. **Nicht gelesen (nur als Beschreibung oder Zusammenfassung):** Molnar, *Interpretable Machine Learning*, Kap. 17 und 18; Covert, Lee 2021 (nur Zusammenfassung; ob dort steht, dass gepaartes Ziehen für niedrige Ordnung exakt ist, ist nicht gelesen); Hwang u. a. 2026 (unbegutachtet). **Eigene Rechnung, nicht aus einer Quelle gelesen:** die Herleitung, warum gepaartes Ziehen bis Ordnung 2 exakt ist, und die Schranke „Ordnung höchstens Baumtiefe“ (gemessen, kein Satz).

## Lokal ausführen

```bash
python3 -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate

pip install -r requirements.txt
streamlit run app.py
```

## Tests ausführen

```bash
pip install -r requirements-dev.txt
pytest tests/ -v
```

Die eingefrorenen Dateien neu erzeugen oder prüfen (Dev-venv, dauert einige Minuten bis eine Viertelstunde je Werkzeug): `python tools/shn_make_costs.py --check`, ebenso `shn_make_learn.py` und `shn_make_games.py`; `--zweit` rechnet mit anderen Zufalls-Seeds.

Gebaut mit Streamlit, Plotly und NumPy.

---

Diese Demo ist Teil des Portfolios von [Sebastian Hanisch](https://sebastianhanisch.net) – Operations Research und Machine Learning ([Über mich](https://sebastianhanisch.net/ueber-mich.html)). Mehr zu den Demos: [Interaktive Demos](https://sebastianhanisch.net/demos.html).
