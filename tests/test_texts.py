"""Zwei Fassungen desselben Textes (App und README) werden verglichen, nicht von Hand synchron gehalten (DEMO-PLAYBOOK §5): die Tabelle „Wo die Annahmen enden“ steht wörtlich in beiden.

Außerdem: Zahlen der festen Textstücke, die die App anzeigt (Negativbefunde, Integrated-Gradients-Absatz), gegen die Messung."""

from pathlib import Path

from streamlit.testing.v1 import AppTest

import shn_experiments as E
import shn_texts as TX

ROOT = Path(__file__).resolve().parent.parent
README = (ROOT / "README.md").read_text(encoding="utf-8")


def test_grenzen_tabelle_steht_wortgleich_in_der_readme_und_in_der_app():
    assert TX.GRENZEN_MARKDOWN.strip() in README
    at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=300)
    at.run()
    assert not at.exception
    assert any(TX.GRENZEN_MARKDOWN.strip() == m.value.strip() for m in at.markdown)
    assert any("📐 Mathematische Formulierung" == e.label for e in at.expander)           # Zugriff über den Titel, nicht über den Index
    assert any(e.label.startswith("So funktioniert") for e in at.expander)


def test_negativbefund_zahlen_stimmen_mit_der_messung():
    assert "81 der 131" in TX.NEGATIV_1 and sum(1 for b in E.formen(0.9) if E.T.wurzel_merkmal(b) == 0) == 81
    assert "64 Koalitionen" in TX.NEGATIV_4 and (1 << 6) - 2 == 62                          # 64 Koalitionen einschließlich leerer und voller
    assert "(Erstmessung) mehr Schritte" in TX.NEGATIV_3 and "nur in 6 von 8" in TX.NEGATIV_3   # Erstmessung 8 von 8, Zweitmessung 6 von 8 (test_claims)
    assert "wenn jede Ziehung neu ausgewertet wird" in TX.NEGATIV_4 and "62 innere Koalitionen" in TX.NEGATIV_4


def test_ig_absatz_zahlen():
    assert "(1, 0)" in TX.IG_ABSATZ and "(1/2, 1/2)" in TX.IG_ABSATZ and "0,357" in TX.IG_ABSATZ and "0,00375" in TX.IG_ABSATZ and "0,00874" in TX.IG_ABSATZ   # Zahlen aus test_integrated_gradients


def test_formeln_haben_kein_ungeschuetztes_prozentzeichen():
    for _, formel in TX.MATHE_BLOECKE:
        assert "%" not in formel                                                             # KaTeX liest % als Kommentar
