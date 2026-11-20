"""Permalink (Begrenzen), Presets und Zufalls-Seed-Button (Standardmuster aus dem Demo-Portfolio), hier mit Auswahllisten.

Es gibt keinen Zufalls-Seed-Regler: alle Ziehungen der Experimente haben feste Zufalls-Seeds (SplitMix64), die Messungen stehen eingefroren in `data/`."""

from dataclasses import dataclass

import streamlit as st

import shn_constants as C


@dataclass(frozen=True)
class SettingSpec:
    url_param: str
    caster: type
    default: object
    lo: float | None = None
    hi: float | None = None
    optionen: tuple | None = None


SETTING_SPECS = {
    "q_slider": SettingSpec("q", float, C.DEFAULT_Q, C.Q_MIN, C.Q_MAX),
    "form_select": SettingSpec("form", str, C.DEFAULT_FORM),
    "fkt_radio": SettingSpec("fkt", str, C.DEFAULT_FUNKTIONAL, optionen=C.FUNKTIONALE),
    "lern_select": SettingSpec("lern", str, C.DEFAULT_LERN, optionen=C.LERN_OPTIONEN),
    "baum_select": SettingSpec("baum", str, C.DEFAULT_BAUM, optionen=C.BAUM_ARTEN),
    "d3_slider": SettingSpec("d3", int, C.DEFAULT_D3, C.D3_MIN, C.D3_MAX),
    "spiel_select": SettingSpec("spiel", str, C.DEFAULT_SPIEL, optionen=C.SPIEL_OPTIONEN),
    "m_select": SettingSpec("m", int, C.DEFAULT_M, optionen=C.M_OPTIONEN),
    "variante_radio": SettingSpec("variante", str, C.DEFAULT_VARIANTE, optionen=C.VARIANTEN),
    "zieh_slider": SettingSpec("zieh", int, C.DEFAULT_ZIEH, C.ZIEH_MIN, C.ZIEH_MAX),
    "ordnung_radio": SettingSpec("ordnung", int, C.DEFAULT_ORDNUNG, optionen=C.ORDNUNGEN),
    "paare_slider": SettingSpec("paare", int, C.DEFAULT_PAARE, C.PAARE_MIN, C.PAARE_MAX),
    "dspiel_radio": SettingSpec("dspiel", int, C.DEFAULT_SPIEL_D, optionen=C.SPIEL_D_OPTIONEN),
}
PRESET_KEYS = {"q": "q_slider", "form": "form_select", "fkt": "fkt_radio", "lern": "lern_select", "baum": "baum_select", "d3": "d3_slider", "spiel": "spiel_select",
               "m": "m_select", "variante": "variante_radio", "zieh": "zieh_slider", "ordnung": "ordnung_radio", "paare": "paare_slider", "dspiel": "dspiel_radio"}


def init_session_state_defaults():
    for state_key, spec in SETTING_SPECS.items():
        if state_key not in st.session_state:
            st.session_state[state_key] = spec.default


def bounds(state_key):
    spec = SETTING_SPECS[state_key]
    return spec.lo, spec.hi


def _begrenzen(spec: SettingSpec, value):
    """Wert auf die Grenzen bzw. die Auswahlliste begrenzen; None, wenn der Wert nicht erlaubt ist."""
    if spec.optionen is not None:
        return value if value in spec.optionen else None
    if spec.lo is not None:
        value = max(spec.lo, value)
    if spec.hi is not None:
        value = min(spec.hi, value)
    return value


def load_permalink_settings():
    """Einmal je Sitzung: Werte aus der Adresszeile übernehmen, auf die Reglergrenzen begrenzt; unbekannte Auswahlwerte werden ignoriert."""
    if "permalink_loaded" in st.session_state:
        return
    qp = st.query_params
    for state_key, spec in SETTING_SPECS.items():
        if spec.url_param in qp:
            try:
                value = spec.caster(qp[spec.url_param])
            except (ValueError, TypeError):
                continue
            value = _begrenzen(spec, value)
            if value is not None:
                st.session_state[state_key] = value
    st.session_state["permalink_loaded"] = True


def sync_query_params(values):
    """`values`: {state_key: aktueller Wert}; die Adresszeile spiegelt die Konfiguration."""
    try:
        for state_key, value in values.items():
            st.query_params[SETTING_SPECS[state_key].url_param] = str(value)
    except Exception:
        pass


def apply_preset(name):
    for key, state_key in PRESET_KEYS.items():
        st.session_state[state_key] = C.PRESETS[name].get(key, SETTING_SPECS[state_key].default)


def aktives_preset():
    """Name des Presets, dessen Werte GENAU den aktuellen Einstellungen entsprechen, sonst None.

    Zustandslos: verglichen wird jeder Regler aus SETTING_SPECS (ein Preset ohne Eintrag für einen Regler meint dessen Standardwert) mit dem Session State."""
    preset_key_of = {state_key: key for key, state_key in PRESET_KEYS.items()}
    for name, preset in C.PRESETS.items():
        if all(
            st.session_state.get(state_key, spec.default) == preset.get(preset_key_of.get(state_key), spec.default)
            for state_key, spec in SETTING_SPECS.items()
        ):
            return name
    return None


def sidebar_presets():
    """Schnellstart oben in der Seitenleiste; das aktive Preset ist hervorgehoben (`type="primary"`)."""
    st.header("🎯 Schnellstart")
    st.caption("Ein Beispielszenario laden:")
    aktiv = aktives_preset()
    for name in C.PRESETS:
        st.button(
            name, width="stretch", on_click=apply_preset, args=(name,), help=C.PRESET_HELP.get(name, ""), key=f"preset_{name}",
            type="primary" if aktiv == name else "secondary",
        )
