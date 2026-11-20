"""Wie wird der Beitrag ausgerechnet? Baum-SHAP und KernelSHAP - interaktive Konzept-Demo
Sebastian Hanisch - Operations Research und Machine Learning

Zwei Rechenwege für Beitragszahlen (Shapley-Werte) werden nachgezählt: Baum-SHAP (pfadabhängiges Funktional des Baums, Aufzählung gegen Algorithmus 2) und KernelSHAP (gezogene Koalitionen,
gepaart oder einfach). Alles an erfundenen Daten (Container-Verweilzeit, binäres Toy); Messungen aus scikit-learn-Modellen sind eingefroren (`data/`), die Rechenwege sind eigene.

Lauffähig mit: streamlit run app.py
"""

import streamlit as st

import shn_constants as C
import shn_evaluation as ev
import shn_experiments as E
import shn_texts as TX
import shn_visualization as V
from shn_layout import dunkles_design, kopf_css
from shn_presets import (
    bounds,
    init_session_state_defaults,
    load_permalink_settings,
    sidebar_presets,
    sync_query_params,
)

st.set_page_config(page_title="Wie wird der Beitrag ausgerechnet? Baum-SHAP und KernelSHAP – Sebastian Hanisch", layout="wide")


@st.cache_data(show_spinner=False)
def _exp1(q, form_index, funktional):
    return E.exp1(q, form_index, funktional)


@st.cache_data(show_spinner=False)
def _exp2(label):
    return E.exp2(label)


@st.cache_data(show_spinner=False)
def _exp3(baum_art, d):
    return E.exp3(baum_art, d)


@st.cache_data(show_spinner=False)
def _kosten_tabelle():
    return E.kosten_tabelle()


@st.cache_data(show_spinner=False)
def _ensembles():
    return E.ensembles()


@st.cache_data(show_spinner=False)
def _kernel(d, modell, m, gepaart, ziehungen):
    return E.kernel_experiment(d, modell, m, gepaart, ziehungen)


@st.cache_data(show_spinner=False)
def _orakel(d, modell):
    return E.orakel_alle_koalitionen(d, modell)


@st.cache_data(show_spinner=False)
def _ordnung():
    return E.ordnung_tabelle()


@st.cache_data(show_spinner=False)
def _gepaart_fehler():
    return E.gepaart_fehler_je_modell()


@st.cache_data(show_spinner=False)
def _spiele(d, ordnung, m):
    return E.zufallsspiele(d, ordnung, m)


@st.cache_data(show_spinner=False)
def _mini():
    return E.mini_beispiel()


def _md_tabelle(kopf, zeilen):
    """Markdown-Tabelle aus Kopfzeile und Zeilen (alles Text, damit das Dezimalkomma überall gleich ist)."""
    out = ["| " + " | ".join(kopf) + " |", "|" + "|".join("---" for _ in kopf) + "|"]
    out += ["| " + " | ".join(str(c) for c in z) + " |" for z in zeilen]
    return "\n".join(out)


def _bruch(x):
    """Bruch als Text (n/m)."""
    return str(x)


# Fester Kopf: nur der Titel bleibt beim Scrollen oben (Stylesheet in shn_layout.py).
st.html(kopf_css(dunkles_design()))
with st.container(key="kopf"):
    st.title("🧮 " + TX.TITEL)
st.markdown(TX.EINLEITUNG)

with st.expander("So funktioniert Baum-SHAP und KernelSHAP", expanded=True):
    st.markdown(TX.SO_FUNKTIONIERT)

st.caption("🔗 Die Adresszeile oben spiegelt Ihre aktuelle Konfiguration wider – einfach kopieren, um ein Szenario zu teilen.")

load_permalink_settings()
init_session_state_defaults()

with st.sidebar:
    sidebar_presets()  # Schnellstart oben, das aktive Preset ist hervorgehoben
    st.header("🌳 Experiment 1: Baumform")
    q = st.slider("Abhängigkeit q (x₂ ist x₁ mit Wahrscheinlichkeit q)", *bounds("q_slider"), step=C.Q_STEP, key="q_slider", format="%.2f",
                  help="Toy: x₁ gleichverteilt, x₂ gleich x₁ mit Wahrscheinlichkeit q, x₃ unabhängig. Bei q = 0,50 ist x₂ von x₁ unabhängig.")
    form = st.selectbox("Baumform (alle berechnen f = x₁)", E.form_labels(), key="form_select",
                        help="Alle 131 Binärbäume, die f = x₁ auf allen acht Punkten berechnen; Kurzschrift: x2[links,rechts] heißt Schnitt auf x₂, Blätter sind die Werte 0 und 1.")
    funktional = st.radio("Funktional", C.FUNKTIONALE, key="fkt_radio", horizontal=True,
                          help="Pfadabhängig: Rechenweg des Baums (Baum-SHAP). Interventional und konditional: am Baummodell, also an seinen Vorhersagen.")
    st.header("🎓 Experiment 2: Lernverfahren")
    lern = st.selectbox("Lernverfahren und Daten", C.LERN_OPTIONEN, key="lern_select",
                        help="Eingefrorene Messungen: Toy (30 Zufalls-Seeds) oder Vehikel mit dem Ziel Vorlauf (5 Zufalls-Seeds, fünf Instanzmengen).")
    st.header("⚖️ Experiment 3: Zwei Rechenwege")
    baum_art = st.selectbox("Baum", C.BAUM_ARTEN, key="baum_select",
                            help="Ein eingefrorener Baum aus scikit-learn (erfundene Zielfunktion über d Merkmalen) oder der vollständige Baum mit Tiefe d über d verschiedenen Merkmalen.")
    d3 = st.slider("Zahl der Merkmale d", *bounds("d3_slider"), key="d3_slider", help="Die Aufzählung rechnet 2^d Koalitionen.")
    st.header("🎲 Experiment 4: KernelSHAP")
    spiel = st.selectbox("Modell und Zahl der Merkmale", C.SPIEL_OPTIONEN, key="spiel_select",
                         help="Eingefrorene Spieltabellen: v(S) für alle 2^d Koalitionen jeder Zeile, interventional mit 30 Hintergrundzeilen.")
    m_koal = st.select_slider("Koalitionen M", options=C.M_OPTIONEN, key="m_select", help="Zahl der gezogenen Koalitionen (bei gepaart die Hälfte Ziehungen, die andere Hälfte Komplemente).")
    variante = st.radio("Ziehen", C.VARIANTEN, key="variante_radio", horizontal=True, help="Einfach: Koalitionen mit Zurücklegen. Gepaart: jede Koalition mit ihrem Komplement.")
    zieh = st.slider("Ziehungen je Instanzmenge", *bounds("zieh_slider"), key="zieh_slider",
                     help="Zufalls-Seeds der Ziehung: je Instanzmenge so viele unabhängige Ziehungen, alle Zeilen der Menge bekommen dieselben Koalitionen.")
    st.header("🧩 Experiment 5: Die Ordnung")
    dspiel = st.radio("Merkmale d des Zufallsspiels", C.SPIEL_D_OPTIONEN, key="dspiel_radio", horizontal=True)
    ordnung_g = st.radio("Ordnung des Zufallsspiels", C.ORDNUNGEN, key="ordnung_radio", horizontal=True,
                         help="Größte Wechselwirkung: 1 additiv, 2 Paare, 3 Dreierterme.")
    paare = st.slider("Gezogene Paare m", *bounds("paare_slider"), key="paare_slider", help="Jedes Paar sind zwei Koalitionen (Koalition und Komplement).")

sync_query_params({"q_slider": float(q), "form_select": form, "fkt_radio": funktional, "lern_select": lern, "baum_select": baum_art, "d3_slider": int(d3),
                   "spiel_select": spiel, "m_select": int(m_koal), "variante_radio": variante, "zieh_slider": int(zieh), "ordnung_radio": int(ordnung_g),
                   "paare_slider": int(paare), "dspiel_radio": int(dspiel)})

# =================================================================================================== Experiment 1
st.markdown("## ❓ Kernfrage 1: " + TX.KERNFRAGE_1)
st.subheader("🔬 Experiment 1: Eine Funktion, viele Bäume")
st.caption(TX.EXP1_CAPTION)
mini = _mini()
with st.expander("Das Mini-Beispiel von Hand: zwei Merkmale, zwei Bäume, derselbe Funktionswert"):
    st.markdown(
        "Verteilung der Punkte (x₁, x₂): (0, 0) 9/20, (1, 1) 9/20, (0, 1) 1/20, (1, 0) 1/20; f = x₁; Zeile x = (1, 1). **Baum A** liest nur x₁, **Baum B** trennt zuerst auf x₂ und dann auf x₁. "
        "Beide berechnen f auf allen vier Punkten.\n\n"
        + _md_tabelle(["Baum", "φ (x₁, x₂) an x = (1, 1), pfadabhängig", "Wichtigkeit (x₁, x₂)"],
                      [["A (nur x₁)", f"({_bruch(mini['A']['phi'][0])}, {_bruch(mini['A']['phi'][1])})", f"({_bruch(mini['A']['wichtigkeit'][0])}, {_bruch(mini['A']['wichtigkeit'][1])})"],
                       ["B (x₂ zuerst)", f"({_bruch(mini['B']['phi'][0])}, {_bruch(mini['B']['phi'][1])})", f"({_bruch(mini['B']['wichtigkeit'][0])}, {_bruch(mini['B']['wichtigkeit'][1])})"]])
        + "\n\nBaum B gibt die konditionalen, Baum A die interventionalen Werte der Funktion f: der Rechenweg bringt das Funktional mit."
    )
r1 = _exp1(float(q), E.form_labels().index(form), funktional)
c1, c2 = st.columns(2)
with c1:
    st.markdown(f"**Gewählte Form:** {r1['label']}")
    reihen = {f"{funktional} (dieser Baum)": (r1["wichtigkeit"], C.COLOR_PFAD if funktional == "pfadabhängig" else C.COLOR_INTERV),
              "interventional (Funktion f)": (r1["wahrheit"]["interventional"], C.COLOR_INTERV), "konditional (Funktion f)": (r1["wahrheit"]["konditional"], C.COLOR_KOND)}
    st.plotly_chart(V.fig_wichtigkeit(reihen, ["x₁", "x₂", "x₃"]), width="stretch", key="chart_exp1_wichtigkeit")
with c2:
    st.markdown(f"**Alle {r1['n_formen']} Baumformen:** Zahl je Beitrag für x₂ ({funktional})")
    st.plotly_chart(V.fig_klassen(r1["klassen"], r1["n_formen"]), width="stretch", key="chart_exp1_klassen")
m1, m2, m3 = st.columns(3)
m1.metric("Beitrag x₂ dieser Baum", ev.de(r1["wichtigkeit"][1]))
m2.metric("Masse mit x₂ vor x₁ im Pfad", ev.de(r1["masse_x2_vor_x1"], 2))
m3.metric(f"Baumformen von {r1['n_formen']} mit x₂-Beitrag ungleich 0", f"{sum(c for w, c in r1['klassen'] if abs(w[1]) > 1e-9)}")
kind, text = ev.verdict_formen(r1)
getattr(st, kind)(text)
st.caption(TX.NEGATIV_1)

st.markdown("---")

# =================================================================================================== Experiment 2
st.subheader("🔬 Experiment 2: Was macht ein Lernverfahren daraus?")
st.caption(TX.EXP2_CAPTION)
r2 = _exp2(lern)
z2 = r2["zeile"]
if r2["welt"] == "toy":
    reihen = {"pfadabhängig (Baummodell)": (z2["pfad"], C.COLOR_PFAD), "interventional (Modell)": (z2["interv"], C.COLOR_INTERV), "konditional (Modell)": (z2["kond"], C.COLOR_KOND)}
    st.plotly_chart(V.fig_wichtigkeit(reihen, ["x₁", "x₂", "x₃"]), width="stretch", key="chart_exp2")
    a, b, c = st.columns(3)
    a.metric("x₂ pfadabhängig (Mittel über Zufalls-Seeds)", ev.de(z2["pfad"][1]))
    b.metric("x₂ interventional (Modell)", ev.de(z2["interv"][1]))
    c.metric(f"Zufalls-Seeds mit |φ₂| > 0,02 (von {z2['n_seeds']})", f"{round(z2['anteil_phi2'] * z2['n_seeds'])}")
else:
    reihen = {"pfadabhängig": ([sum(m[j] for m in z2["pfad_je_menge"]) / len(z2["pfad_je_menge"]) for j in range(5)], C.COLOR_PFAD),
              "interventional": ([sum(m[j] for m in z2["interv_je_menge"]) / len(z2["interv_je_menge"]) for j in range(5)], C.COLOR_INTERV)}
    st.plotly_chart(V.fig_wichtigkeit(reihen, ["Art", "Kunde", "Zoll", "Wochenende", "Vorlauf"]), width="stretch", key="chart_exp2")
    st.caption("Säulen: Mittel über die fünf Instanzmengen zu je 25 Zeilen und 5 Zufalls-Seeds; die Spanne über die Mengen steht im Verdikt.")
kind, text = ev.verdict_lernen(r2)
getattr(st, kind)(text)
st.caption(TX.NEGATIV_2)

st.markdown("---")

# =================================================================================================== Experiment 3
st.markdown("## ❓ Kernfrage 2: " + TX.KERNFRAGE_2)
st.subheader("🔬 Experiment 3: Baum-SHAP, zwei Rechenwege, Kosten und Abstand")
st.caption(TX.EXP3_CAPTION)
r3 = _exp3(baum_art, int(d3))
tabelle = _kosten_tabelle()
a, b, c, d_ = st.columns(4)
a.metric("Blätter", r3["blaetter"])
b.metric("Aufzählung (Besuche)", ev.tausend(r3["besuche"]))
c.metric("Algorithmus 2 (Schritte)", ev.tausend(r3["schritte"]))
d_.metric("größte Abweichung der Beiträge", f"{r3['abweichung']:.1e}".replace(".", ","))
st.plotly_chart(V.fig_kosten(tabelle, r3), width="stretch", key="chart_exp3")
kind, text = ev.verdict_kosten_live(r3)
getattr(st, kind)(text)
with st.expander("Kostentabelle (eingefroren, acht Bäume je Zelle, vier bei d = 14): Schritte je Instanz"):
    st.markdown(_md_tabelle(
        ["d", "Tiefe", "Blätter", "Aufzählung (Mittel, Spanne)", "Algorithmus 2 (Mittel, Spanne)", "Verhältnis Aufzählung zu Algorithmus 2", "Bäume mit mehr Schritten bei Algorithmus 2"],
        [[z["d"], z["tiefe"], f"{z['blaetter'][0]} bis {z['blaetter'][1]}", f"{ev.tausend(round(z['aufzaehlung'][0]))} ({ev.tausend(z['aufzaehlung'][1])} bis {ev.tausend(z['aufzaehlung'][2])})",
          f"{ev.tausend(round(z['alg2'][0]))} ({ev.tausend(z['alg2'][1])} bis {ev.tausend(z['alg2'][2])})", ev.spanne(z["verhaeltnis"][0], z["verhaeltnis"][1], 2),
          f"{z['alg2_mehr']} von {z['n_baeume']}"] for z in tabelle]))
kind, text = ev.verdict_kosten_tabelle(tabelle)
getattr(st, kind)(text)
ens = _ensembles()
st.markdown("**Am trainierten Ensemble:** wie weit liegt der pfadabhängige Beitrag vom interventionalen entfernt? (eingefroren, Hintergrund = exakte Bevölkerung des Generators)")
st.markdown(_md_tabelle(
    ["Modell (R² gegen μ)", "Abstand, Median je Menge (Spanne)", "Abstand, 90 %-Quantil (Spanne)", "anderes Rang 1 (Spanne)", "anderes Vorzeichen des größten Beitrags (Erstmessung)"],
    [[f"{z['modell']} ({ev.de(z['r2'], 3)})", ev.spanne(100 * z["median"][0], 100 * z["median"][1], 1) + " %", ev.spanne(100 * z["q90"][0], 100 * z["q90"][1], 1) + " %",
      ev.spanne(100 * z["rang1"][0], 100 * z["rang1"][1], 1) + " %", ev.spanne(100 * z["vorzeichen"][0], 100 * z["vorzeichen"][1], 1) + " %"] for z in ens]))
kind, text = ev.verdict_ensembles(ens)
getattr(st, kind)(text)
st.caption(TX.NEGATIV_3)

st.markdown("---")

# =================================================================================================== Experiment 4
st.markdown("## ❓ Kernfrage 3: " + TX.KERNFRAGE_3)
st.subheader("🔬 Experiment 4: KernelSHAP, Kosten gegen Fehler")
st.caption(TX.EXP4_CAPTION)
d4, modell4 = int(spiel.split(" · ")[0].replace("d = ", "")), spiel.split(" · ")[1]
est = int(zieh) * 5 * int(m_koal) * 2.5e-5 + int(zieh) * 5 * (10 if d4 == 6 else 6) * 3e-4
if est > 1.5:
    with st.spinner(f"Zieht {int(zieh) * 5} Ziehungen mit je {int(m_koal)} Koalitionen und löst die Regressionen (etwa {est:.0f} s)"):
        r4 = _kernel(d4, modell4, int(m_koal), variante == "gepaart", int(zieh))
else:
    r4 = _kernel(d4, modell4, int(m_koal), variante == "gepaart", int(zieh))
a, b, c, d_ = st.columns(4)
a.metric("Modellauswertungen", ev.tausend(r4["auswertungen"]), delta=f"{ev.prozent(r4['anteil'])} der Aufzählung ({ev.tausend(r4['auswertungen_aufzaehlung'])})", delta_color="off")
a_lo, a_hi = min(r4["median_je_menge"]), max(r4["median_je_menge"])
b.metric("Medianfehler (Spanne über 5 Mengen)", ev.spanne(a_lo, a_hi))
c.metric("90 %-Quantil des Fehlers", ev.de(r4["q90"]))
d_.metric("Rang 1 wie Aufzählung", ev.spanne(100 * min(r4["rang1_je_menge"]), 100 * max(r4["rang1_je_menge"]), 0) + " %")
st.plotly_chart(V.fig_kernel(r4), width="stretch", key="chart_exp4")
kind, text = ev.verdict_kernel(r4)
getattr(st, kind)(text)
orak = _orakel(d4, modell4)
st.caption(f"Orakel: KernelSHAP mit allen Koalitionen und Shapley-Gewicht gegen die Aufzählung (Lundberg/Lee 2017, Satz 2), größte Abweichung {f'{orak:.1e}'.replace('.', ',')} an diesem Modell.")
st.caption(TX.NEGATIV_4)

st.markdown("---")

# =================================================================================================== Experiment 5
st.subheader("🔬 Experiment 5: Die Ordnung entscheidet")
st.caption(TX.EXP5_CAPTION)
ord_tab = _ordnung()
fehler = _gepaart_fehler()
st.markdown("**Möbius-Ordnung des Spiels trainierter Modelle** (d = 6, interventional, Hintergrund 30 Zeilen; fünf Instanzmengen zu 10 Zeilen) und der Fehler des gepaarten Ziehens bei 24 von 64 Koalitionen:")
st.plotly_chart(V.fig_ordnung(ord_tab), width="stretch", key="chart_exp5_ordnung")
st.markdown(_md_tabelle(
    ["Modell", "größte Ordnung", "Anteil Ordnung 1 / 2 / 3 / 4 / 5 bis 6", "gepaarter Medianfehler bei M = 24 (Spanne über 5 Mengen)"],
    [[z["modell"], z["ordnung"], " / ".join(ev.de(a, 3) for a in (*z["anteile"][:4], sum(z["anteile"][4:]))), ev.spanne(*fehler[z["modell"]])] for z in ord_tab]))
kind, text = ev.verdict_ordnung(ord_tab, fehler)
getattr(st, kind)(text)
st.markdown(f"**Zufallsspiele:** Möbius-Koeffizienten N(0, 1) bis Ordnung {int(ordnung_g)}, d = {int(dspiel)}; gepaartes Ziehen mit {int(paare)} Paaren, Größen aus dem Shapley-Kern.")
with st.spinner("Rechnet 200 Zufallsspiele (erste Rechnung dieser Ordnung, danach zwischengespeichert)"):
    r5 = _spiele(int(dspiel), int(ordnung_g), int(paare))
st.plotly_chart(V.fig_zufallsspiele(r5), width="stretch", key="chart_exp5_spiele")
kind, text = ev.verdict_zufallsspiele(r5)
getattr(st, kind)(text)
st.caption(TX.NEGATIV_5)
with st.expander("Handwert: Ordnung 3 bei d = 4 (Paare der Größen 1 und 3)"):
    st.markdown("Das Spiel mit c₀₁₂ = 1 (nur die Menge {0, 1, 2} wirkt, also v(S) = 1, wenn {0, 1, 2} ⊆ S) hat den Shapley-Wert (1/3, 1/3, 1/3, 0). Gepaartes Ziehen der Koalitionen der Größen 1 und 3 gibt (3/8, 3/8, 3/8, −1/8): "
                "das unbeteiligte Merkmal bekommt −1/8. Alle 14 inneren Koalitionen ungewichtet geben (5/16, 5/16, 5/16, 1/16), ebenfalls nicht exakt: für Ordnung 3 braucht man das Gewicht des Shapley-Kerns, nicht nur Vollständigkeit.")

st.markdown("---")
st.markdown(TX.IG_ABSATZ)

st.markdown("---")

# =================================================================================================== Grenzen und Formeln
st.subheader("🚧 Wo die Annahmen enden")
st.markdown(TX.GRENZEN_MARKDOWN)

st.markdown("---")

with st.expander("📐 Mathematische Formulierung"):
    st.markdown(TX.MATHE_LESEHILFE)
    for text_, formel in TX.MATHE_BLOECKE:
        st.markdown(text_)
        st.latex(formel)
    st.markdown(TX.MATHE_SCHLUSS)
    st.markdown(TX.LITERATUR)

st.markdown("---")
st.caption(
    "Diese Demo ist Teil des Portfolios von [Sebastian Hanisch](https://sebastianhanisch.net) – "
    "Operations Research und Machine Learning ([Über mich](https://sebastianhanisch.net/ueber-mich.html)). "
    "Mehr zu den Demos: [Interaktive Demos](https://sebastianhanisch.net/demos.html)."
)
