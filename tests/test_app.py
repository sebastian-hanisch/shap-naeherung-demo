"""End-to-end Smoke-Test via Streamlits offiziellem AppTest-Framework: lädt app.py mit den Standardeinstellungen, klickt jeden Preset-Button, fährt jeden Regler an seine Grenzen
und durchläuft die Auswahllisten (Lernverfahren, Spieltabellen, Baumarten, Funktionale, Zufallsspiele) und prüft, dass kein Python-Fehler auftritt - insbesondere
`streamlit.errors.StreamlitDuplicateElementId` (mehrere st.plotly_chart-Aufrufe ohne eindeutiges key=) und `StreamlitAPIException` bei Reglern mit berechneten Grenzen."""

import os

import pytest
from streamlit.testing.v1 import AppTest

import shn_constants as C
import shn_experiments as E

APP_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "app.py")
TIMEOUT = int(os.environ.get("APPTEST_TIMEOUT", "300"))  # unter Last (parallele Läufe) reichen 180 s nicht immer; Umgebungsvariable zum Anpassen
AUTOPLAY_MARKER = "▶"  # Auto-Play-Buttons (Animation mit sleep) im Smoke-Test auslassen


def _fresh_app() -> AppTest:
    at = AppTest.from_file(APP_PATH)
    at.run(timeout=TIMEOUT)
    assert not at.exception, [str(e) for e in at.exception]
    return at


def _texte(at):
    return " ".join(x.value for art in (at.success, at.info, at.warning) for x in art)


def test_app_loads_without_exception():
    at = _fresh_app()
    assert len(at.success) + len(at.info) + len(at.warning) >= 7                       # sieben Verdikte (Experimente 1 bis 5, zwei in Experiment 3)


def test_every_button_click_does_not_raise():
    labels = [b.label for b in _fresh_app().button]
    assert labels, "keine Buttons gefunden"
    for index, label in enumerate(labels):
        if AUTOPLAY_MARKER in label:
            continue
        at = _fresh_app()
        at.button[index].click().run(timeout=TIMEOUT)
        assert not at.exception, (label, [str(e) for e in at.exception])


def test_every_slider_at_its_min_and_max_does_not_raise():
    labels = [s.label for s in _fresh_app().slider]
    assert labels, "keine Slider gefunden"
    for index, label in enumerate(labels):
        for edge in ("min", "max"):
            at = _fresh_app()
            slider = at.slider[index]
            value = slider.min if edge == "min" else slider.max
            if isinstance(slider.value, int):
                value = int(value)
            slider.set_value(value).run(timeout=TIMEOUT)
            assert not at.exception, (label, edge, [str(e) for e in at.exception])


@pytest.mark.parametrize("key,optionen", [("lern_select", C.LERN_OPTIONEN), ("spiel_select", C.SPIEL_OPTIONEN), ("baum_select", C.BAUM_ARTEN), ("fkt_radio", C.FUNKTIONALE),
                                          ("variante_radio", C.VARIANTEN), ("ordnung_radio", C.ORDNUNGEN), ("dspiel_radio", C.SPIEL_D_OPTIONEN), ("m_select", C.M_OPTIONEN)])
def test_jede_auswahl_laeuft_ohne_fehler(key, optionen):
    at = _fresh_app()
    for o in optionen:
        regler = next(w for art in ("selectbox", "radio", "select_slider") for w in getattr(at.sidebar, art) if w.key == key)
        regler.set_value(o).run(timeout=TIMEOUT)
        assert not at.exception, (key, o, [str(e) for e in at.exception])


def test_baumformen_stichprobe_laeuft_ohne_fehler():
    at = _fresh_app()
    labels = E.form_labels()
    for i in (0, 1, 5, 40, 90, 130):
        next(w for w in at.sidebar.selectbox if w.key == "form_select").set_value(labels[i]).run(timeout=TIMEOUT)
        assert not at.exception, (labels[i], [str(e) for e in at.exception])


def test_d3_und_baumart_kombinationen_an_den_grenzen():
    at = _fresh_app()
    for baum in C.BAUM_ARTEN:
        for d in (C.D3_MIN, C.D3_MAX):
            next(w for w in at.sidebar.selectbox if w.key == "baum_select").set_value(baum).run(timeout=TIMEOUT)
            next(w for w in at.sidebar.slider if w.key == "d3_slider").set_value(d).run(timeout=TIMEOUT)
            assert not at.exception, (baum, d, [str(e) for e in at.exception])
            assert f"d = {d}" in _texte(at)


def test_viele_ziehungen_und_grosses_m_laufen_mit_spinner_ohne_fehler():
    at = _fresh_app()
    next(w for w in at.sidebar.select_slider if w.key == "m_select").set_value(1280).run(timeout=TIMEOUT)
    next(w for w in at.sidebar.selectbox if w.key == "spiel_select").set_value("d = 10 · Boosting Tiefe 3").run(timeout=TIMEOUT)
    next(w for w in at.sidebar.radio if w.key == "variante_radio").set_value("einfach").run(timeout=TIMEOUT)
    next(w for w in at.sidebar.slider if w.key == "zieh_slider").set_value(20).run(timeout=TIMEOUT)
    assert not at.exception
    assert "M = 1280 Koalitionen" in _texte(at) and "einfach" in _texte(at)
