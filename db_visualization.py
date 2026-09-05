"""Plotly-Visualisierungen: Punktwolke mit Core/Border/Noise-Einfärbung (Kernvisual,
schrittanimierbar wie in branch-bound-demo/kmeans-demo), k-Distanz-Plot und
eps-Sweep-Chart (gesamt + je wahrer Gruppe) für die "Wie wählt man eps?"-Sektion."""

import numpy as np

from db_algorithm import NOISE, UNVISITED, labels_at_step
from db_evaluation import roles_at_step

CLUSTER_PALETTE = [
    "#1f77b4", "#d68a2e", "#2ca02c", "#d62728",
    "#9467bd", "#8c564b", "#e377c2", "#17becf",
]
NOISE_COLOR = "#9aa6ba"
UNVISITED_COLOR = "#e3e7ee"


def _axis_range(data):
    xmin, xmax = data[:, 0].min(), data[:, 0].max()
    ymin, ymax = data[:, 1].min(), data[:, 1].max()
    padx = (xmax - xmin) * 0.1 or 1.0
    pady = (ymax - ymin) * 0.1 or 1.0
    return [xmin - padx, xmax + padx], [ymin - pady, ymax + pady]


def _add_role_traces(fig, data, labels, roles, legend=True):
    roles = np.array(roles, dtype=object)

    mask_unvisited = labels == UNVISITED
    if mask_unvisited.any():
        fig.add_trace(
            go_scatter(data, mask_unvisited, "Noch nicht besucht", UNVISITED_COLOR, size=6, legend=legend)
        )

    mask_noise = labels == NOISE
    if mask_noise.any():
        fig.add_trace(
            go_scatter(data, mask_noise, "Noise", NOISE_COLOR, size=8, symbol="x", legend=legend)
        )

    for cid in sorted(int(l) for l in set(labels.tolist()) if l >= 0):
        color = CLUSTER_PALETTE[cid % len(CLUSTER_PALETTE)]
        mask_core = (labels == cid) & (roles == "core")
        mask_border = (labels == cid) & (roles == "border")
        if mask_core.any():
            fig.add_trace(
                go_scatter(data, mask_core, f"Cluster {cid + 1}", color, size=8, legendgroup=f"c{cid}", legend=legend)
            )
        if mask_border.any():
            fig.add_trace(
                go_scatter(
                    data, mask_border, f"Cluster {cid + 1} (Rand)", color, size=8, symbol="circle-open",
                    line_width=2, legendgroup=f"c{cid}", legend=False,
                )
            )


def go_scatter(data, mask, name, color, size, symbol="circle", line_width=0.5, legendgroup=None, legend=True):
    import plotly.graph_objects as go

    return go.Scatter(
        x=data[mask, 0], y=data[mask, 1], mode="markers", name=name, showlegend=legend,
        legendgroup=legendgroup,
        marker=dict(color=color, size=size, symbol=symbol, line=dict(width=line_width, color="white" if symbol == "circle" else color)),
        hoverinfo="skip",
    )


def build_scatter_figure(instance, result, step):
    import plotly.graph_objects as go

    data = np.array(instance.points)
    labels = labels_at_step(instance.n_points, result.events, step)
    roles = roles_at_step(instance.n_points, result.events, step)

    fig = go.Figure()
    _add_role_traces(fig, data, labels, roles, legend=True)

    xr, yr = _axis_range(data)
    fig.update_layout(
        template="plotly_white", height=460,
        xaxis=dict(visible=False, range=xr, fixedrange=True),
        yaxis=dict(visible=False, range=yr, fixedrange=True, scaleanchor="x", scaleratio=1),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
        margin=dict(t=40, l=10, r=10, b=10),
    )
    return fig


def build_mini_scatter_figure(instance, result):
    """Kompakte, legendenlose Variante fuer die Kleinmultiples ("und mit anderen
    eps-Werten?") - zeigt immer nur das Endergebnis eines Laufs."""
    import plotly.graph_objects as go

    data = np.array(instance.points)
    labels = np.array(result.final_labels)
    roles = result.final_roles

    fig = go.Figure()
    _add_role_traces(fig, data, labels, roles, legend=False)

    xr, yr = _axis_range(data)
    fig.update_layout(
        template="plotly_white", height=200,
        xaxis=dict(visible=False, range=xr, fixedrange=True),
        yaxis=dict(visible=False, range=yr, fixedrange=True, scaleanchor="x", scaleratio=1),
        margin=dict(t=5, l=5, r=5, b=5), showlegend=False,
    )
    return fig


def build_k_distance_chart(k_distances, min_samples):
    import plotly.graph_objects as go

    k = max(1, min_samples - 1)
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=list(range(1, len(k_distances) + 1)), y=list(k_distances), mode="lines",
            line=dict(color="#1f77b4", width=2), name=f"{k}-Distanz",
        )
    )
    fig.update_layout(
        template="plotly_white", height=280,
        xaxis_title="Punkte (aufsteigend nach Distanz sortiert)", yaxis_title=f"Distanz zum {k}-ten Nachbarn",
        showlegend=False, margin=dict(t=20, l=10, r=10, b=10),
    )
    fig.update_xaxes(fixedrange=True)
    fig.update_yaxes(fixedrange=True)
    return fig


def build_eps_sweep_chart(sweep):
    import plotly.graph_objects as go

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=sweep.eps_values, y=[v * 100 for v in sweep.noise_fraction], mode="lines", name="Noise gesamt",
            line=dict(color="#14233B", width=2, dash="dot"),
        )
    )
    for group in sorted(sweep.per_group_noise_fraction.keys()):
        fractions = sweep.per_group_noise_fraction[group]
        color = CLUSTER_PALETTE[group % len(CLUSTER_PALETTE)]
        fig.add_trace(
            go.Scatter(
                x=sweep.eps_values, y=[v * 100 for v in fractions], mode="lines", name=f"Noise Gruppe {group + 1}",
                line=dict(color=color, width=2),
            )
        )
    fig.update_layout(
        template="plotly_white", height=300,
        xaxis_title="eps", yaxis_title="Noise-Anteil (%)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
        margin=dict(t=30, l=10, r=10, b=10),
    )
    fig.update_xaxes(fixedrange=True)
    fig.update_yaxes(fixedrange=True, range=[0, 100])
    return fig
