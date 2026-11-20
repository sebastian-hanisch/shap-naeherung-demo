"""Plotly-Abbildungen. Alle Achsen sind gesperrt (fixedrange), damit Touch-Geräte beim Scrollen nicht am Diagramm hängen bleiben."""

import plotly.graph_objects as go

import shn_constants as C


def lock_axes(fig):
    fig.update_xaxes(fixedrange=True)
    fig.update_yaxes(fixedrange=True)
    return fig


def _base(fig, height=340, legend=True):
    fig.update_layout(height=height, margin=dict(l=10, r=10, t=10, b=10), plot_bgcolor="rgba(0,0,0,0)", showlegend=legend,
                      legend=dict(orientation="h", y=-0.25))
    return lock_axes(fig)


def fig_wichtigkeit(reihen: dict, merkmale: list, titel_y: str = "Wichtigkeit (Mittel von |φ|)"):
    """Gruppierte Säulen: je Merkmal eine Gruppe, je Reihe (Funktional oder Rechenweg) eine Farbe. `reihen`: Name → (Werte, Farbe)."""
    fig = go.Figure()
    for nm, (werte, farbe) in reihen.items():
        fig.add_trace(go.Bar(x=merkmale, y=list(werte), name=nm, marker_color=farbe, hovertemplate="%{x}: %{y:.3f}<extra>" + nm + "</extra>"))
    fig.update_layout(barmode="group")
    fig.update_yaxes(title_text=titel_y, rangemode="tozero")
    return _base(fig)


def fig_klassen(klassen, n_formen: int):
    """Säulen: Zahl der Baumformen je Wert des Beitrags von x₂ (pfadabhängig)."""
    xs = [f"x₂: {w[1]:.3f}".replace(".", ",") for w, _ in klassen]
    ys = [c for _, c in klassen]
    fig = go.Figure(go.Bar(x=xs, y=ys, marker_color=[C.COLOR_INTERV if abs(w[1]) < 1e-9 else C.COLOR_PFAD for w, _ in klassen], text=ys, textposition="outside",
                           hovertemplate="%{x}: %{y} von " + str(n_formen) + " Baumformen<extra></extra>"))
    fig.update_yaxes(title_text="Baumformen", range=[0, n_formen * 1.1])
    fig.update_xaxes(title_text="Beitrag für das nie gelesene x₂")
    return _base(fig, 300, legend=False)


def fig_kosten(zeilen, live=None):
    """Schritte gegen d (logarithmisch): je Tiefe zwei Linien (Aufzählung, Algorithmus 2), Mittel über die Bäume einer Zelle; live: markierter Punkt des gewählten Baums."""
    fig = go.Figure()
    stile = {3: "solid", 6: "dash", 8: "dot"}
    for tiefe in (3, 6, 8):
        z = sorted((r for r in zeilen if r["tiefe"] == tiefe), key=lambda r: r["d"])
        ds = [r["d"] for r in z]
        fig.add_trace(go.Scatter(x=ds, y=[r["aufzaehlung"][0] for r in z], mode="lines+markers", name=f"Aufzählung, Tiefe {tiefe}", line=dict(color=C.COLOR_ALG1, dash=stile[tiefe])))
        fig.add_trace(go.Scatter(x=ds, y=[r["alg2"][0] for r in z], mode="lines+markers", name=f"Algorithmus 2, Tiefe {tiefe}", line=dict(color=C.COLOR_ALG2, dash=stile[tiefe])))
    if live:
        fig.add_trace(go.Scatter(x=[live["d"], live["d"]], y=[live["besuche"], live["schritte"]], mode="markers", name="gewählter Baum (Aufzählung, Algorithmus 2)",
                                 marker=dict(size=13, color=[C.COLOR_ALG1, C.COLOR_ALG2], line=dict(width=2, color="black"), symbol="diamond")))
    fig.update_yaxes(title_text="Schritte je Instanz (Zählregel in der Anzeige)", type="log")
    fig.update_xaxes(title_text="Zahl der Merkmale d", dtick=2)
    return _base(fig, 380)


def fig_kernel(r):
    """Medianfehler je Instanzmenge (fünf getrennte Mengen) mit dem Mittelwert über die Mengen als Linie."""
    xs = [f"Menge {i + 1}" for i in range(len(r["median_je_menge"]))]
    fig = go.Figure(go.Bar(x=xs, y=r["median_je_menge"], marker_color=C.COLOR_KOND, hovertemplate="%{x}: %{y:.4f}<extra></extra>", name="Medianfehler"))
    fig.update_yaxes(title_text="Medianfehler (relativ zum größten |φ|)", rangemode="tozero")
    return _base(fig, 300, legend=False)


def fig_ordnung(zeilen):
    """Anteil der Quadratsumme der Möbius-Koeffizienten je Ordnung, ein Balken je Modell."""
    fig = go.Figure()
    farben = ["#2F6B65", "#4c78a8", "#d8a03d", "#C0504D", "#8c6d31", "#7a5195"]
    for o in range(6):
        fig.add_trace(go.Bar(y=[z["modell"] for z in zeilen], x=[z["anteile"][o] for z in zeilen], name=f"Ordnung {o + 1}", orientation="h", marker_color=farben[o],
                             hovertemplate="%{y}: %{x:.3f}<extra>Ordnung " + str(o + 1) + "</extra>"))
    fig.update_layout(barmode="stack")
    fig.update_xaxes(title_text="Anteil an der Quadratsumme der Koeffizienten", range=[0, 1])
    fig.update_yaxes(autorange="reversed")
    return _base(fig, 330)


def fig_zufallsspiele(r):
    xs = [f"Gruppe {i + 1}" for i in range(len(r["anteile"]))]
    fig = go.Figure(go.Bar(x=xs, y=[100 * a for a in r["anteile"]], marker_color=C.COLOR_INTERV, hovertemplate="%{x}: %{y:.0f} %<extra></extra>"))
    fig.update_yaxes(title_text="exakte Spiele (%)", range=[0, 105])
    return _base(fig, 280, legend=False)
