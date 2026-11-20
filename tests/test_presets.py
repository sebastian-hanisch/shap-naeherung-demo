"""Presets: Vollständigkeit, gültige Werte, Standard-Preset = Standardeinstellungen, verschiedene Zustände; Hilfetexte als Behauptungen (jede Zahl hier nachgerechnet); Permalink-Begrenzung."""

import pytest

import shn_constants as C
import shn_data as DATA
import shn_experiments as E
import shn_presets as P
from shn_treeshap import besuche_vollstaendiger_baum, schritte_vollstaendiger_baum


def test_every_preset_has_help_and_only_known_keys():
    assert set(C.PRESETS) == set(C.PRESET_HELP)
    for name, preset in C.PRESETS.items():
        assert set(preset) <= set(P.PRESET_KEYS), name
        assert C.PRESET_HELP[name]


def test_preset_values_are_inside_the_regler_bounds_and_lists():
    for name, preset in C.PRESETS.items():
        for key, value in preset.items():
            spec = P.SETTING_SPECS[P.PRESET_KEYS[key]]
            if spec.optionen is not None:
                assert value in spec.optionen, (name, key)
            elif key == "form":
                assert value in E.form_labels(), name
            else:
                assert spec.lo <= value <= spec.hi, (name, key)
    assert C.FORM_NUR_X1 in E.form_labels() and C.FORM_X2_ZUERST in E.form_labels() and C.DEFAULT_FORM in E.form_labels()


def test_default_preset_is_first_and_equals_the_default_state():
    assert next(iter(C.PRESETS)) == "f = x₁, Baum nur x₁" and C.PRESETS["f = x₁, Baum nur x₁"] == {}
    assert P.SETTING_SPECS["form_select"].default == C.FORM_NUR_X1 and P.SETTING_SPECS["lern_select"].default in C.LERN_OPTIONEN
    assert P.SETTING_SPECS["spiel_select"].default in C.SPIEL_OPTIONEN and P.SETTING_SPECS["m_select"].default in C.M_OPTIONEN


def test_setting_specs_cover_every_preset_key_and_url_params_are_distinct():
    assert set(P.PRESET_KEYS.values()) == set(P.SETTING_SPECS)
    assert len({s.url_param for s in P.SETTING_SPECS.values()}) == len(P.SETTING_SPECS)


def test_presets_unterscheiden_sich_im_zustand():
    zustaende = []
    for preset in C.PRESETS.values():
        zustaende.append(tuple(sorted((k, str(preset.get(k, P.SETTING_SPECS[sk].default))) for k, sk in P.PRESET_KEYS.items())))
    assert len(set(zustaende)) == len(C.PRESETS)


def test_lern_und_spieloptionen_kommen_aus_den_daten():
    assert len(C.LERN_OPTIONEN) == 23 and len(C.SPIEL_OPTIONEN) == 10
    assert set(C.LERN_OPTIONEN) == set(DATA.lern_zeilen()) and set(C.SPIEL_OPTIONEN) == set(DATA.spiel_modelle())


# ------------------------------------------------------------------------------------------------ Hilfetexte als Behauptungen
def test_hilfetext_wald_pfadabhaengig_etwa_0_1():
    r = E.exp2(C.LERN_WALD)
    assert round(r["zeile"]["pfad"][1], 1) == 0.1 and r["zeile"]["interv"][1] < 0.001       # „pfadabhängig etwa 0,1, interventional etwa 0“
    assert r["zeile"]["abweichung_von_f"] == pytest.approx(0.0167, abs=1e-4)               # „f bis auf höchstens 0,0167“


def test_hilfetext_x2_zuerst_gibt_ein_fuenftel():
    r = E.exp1(0.9, E.form_labels().index(C.FORM_X2_ZUERST), "pfadabhängig")
    assert r["wichtigkeit"][1] == pytest.approx(0.2, abs=1e-12) and E.exp1(0.9, 0, "pfadabhängig")["wichtigkeit"][1] == 0.0


def test_hilfetext_boosting_tiefe_2_gepaart_exakt_und_tiefe_4_nicht():
    r2 = E.kernel_experiment(6, "Boosting Tiefe 2", 24, True, 10)
    r4 = E.kernel_experiment(6, "Boosting Tiefe 4", 24, True, 10)
    assert max(r2["median_je_menge"]) < 1e-9 and r2["auswertungen"] == 720
    assert min(r4["median_je_menge"]) > 0.005


def test_hilfetext_netz_d10_40_koalitionen_3_9_prozent():
    r = E.kernel_experiment(10, "Netz", 40, True, 6)
    assert r["auswertungen"] == 1200 and r["auswertungen_aufzaehlung"] == 30720 and round(100 * r["anteil"], 1) == 3.9 and (1 << 10) == 1024


def test_hilfetext_vollstaendiger_baum_d10():
    r = E.exp3("vollständig (Tiefe = d)", 10)
    assert (r["besuche"], r["schritte"]) == (175099, 122881) == (besuche_vollstaendiger_baum(10), schritte_vollstaendiger_baum(10))


# ------------------------------------------------------------------------------------------------ Permalink
def test_permalink_begrenzt_zahlen_und_ignoriert_unbekannte_auswahl():
    spec_q, spec_m, spec_d = P.SETTING_SPECS["q_slider"], P.SETTING_SPECS["m_select"], P.SETTING_SPECS["d3_slider"]
    assert P._begrenzen(spec_q, 7.0) == C.Q_MAX and P._begrenzen(spec_q, -1.0) == C.Q_MIN
    assert P._begrenzen(spec_d, 99) == C.D3_MAX and P._begrenzen(spec_d, 0) == C.D3_MIN
    assert P._begrenzen(spec_m, 40) == 40 and P._begrenzen(spec_m, 41) is None
    assert P._begrenzen(P.SETTING_SPECS["variante_radio"], "gepaart") == "gepaart" and P._begrenzen(P.SETTING_SPECS["variante_radio"], "zweifach") is None
