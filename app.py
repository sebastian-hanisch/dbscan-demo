"""DBSCAN für Sammel-Routen ohne feste Anzahl - interaktive Konzept-Demo
Sebastian Hanisch - Operations Research und Machine Learning

Anders als die Fall-Demos im Portfolio (ein Anwendungsfall, mehrere Verfahren im
Vergleich) zeigt diese Demo EIN Verfahren - DBSCAN - und lässt stattdessen die
Parameterempfindlichkeit wachsen: von gleichmäßig dichten Gruppen, bei denen fast jedes
eps funktioniert, bis zu Gruppen sehr unterschiedlicher Dichte, bei denen kein einzelnes
eps alle Gruppen gleich gut bedient. Drittes Stück der "Konzepte"-Reihe, zweites der
Clustering-Linie nach kmeans-demo (siehe README für die Einordnung).

Lauffähig mit: streamlit run app.py
"""

import time

import streamlit as st

import db_constants as C
from db_algorithm import labels_at_step, run
from db_evaluation import (
    eps_sweep,
    k_distance_values,
    per_group_noise_fraction,
    roles_at_step,
    stats_at_step,
    suggested_eps_range,
)
from db_presets import (
    apply_preset,
    bounds,
    init_session_state_defaults,
    load_permalink_settings,
    randomize_seed,
    sync_query_params,
)
from db_scenario import generate_instance
from db_visualization import build_eps_sweep_chart, build_k_distance_chart, build_mini_scatter_figure, build_scatter_figure

st.set_page_config(page_title="DBSCAN – Sebastian Hanisch", layout="wide")


@st.cache_data(show_spinner=False)
def _compute_run(n_points, k, spread, density_imbalance, outlier_fraction, shape, seed, eps, min_samples):
    instance = generate_instance(n_points, k, spread, density_imbalance, outlier_fraction, shape, seed)
    result = run(instance.as_array(), eps, min_samples)
    return instance, result


@st.cache_data(show_spinner=False)
def _compute_example_run(instance, eps, min_samples):
    return run(instance.as_array(), eps, min_samples)


@st.cache_data(show_spinner=False)
def _compute_eps_tools(instance, min_samples):
    data = instance.as_array()
    k_distances = k_distance_values(data, min_samples)
    eps_values = suggested_eps_range(data, min_samples, C.EPS_SWEEP_STEPS)
    sweep = eps_sweep(data, min_samples, eps_values, true_labels=instance.true_labels)
    return k_distances, sweep


st.title("🧭 DBSCAN: Sammel-Routen ohne feste Anzahl bilden")
st.markdown(
    """
Lieferadressen sollen zu **Sammel-Routen** gruppiert werden, ohne die Anzahl der Routen
vorher festzulegen - dicht beieinanderliegende Adressen bilden automatisch eine Route,
einzelne, zu weit entfernte Adressen werden explizit als **nicht routentauglich**
markiert, statt zwangsweise irgendeiner Route zugeordnet zu werden. Genau das leistet
**DBSCAN**: ein dichtebasiertes Verfahren, das Gruppen direkt aus der lokalen Dichte der
Punkte herleitet, statt - wie k-Means - eine vorher festgelegte Anzahl k von Zentren zu
verteilen. Genau **wie** das funktioniert, erklärt der aufgeklappte Abschnitt direkt
darunter - bevor weiter unten die Suche live dazu läuft und die Frage
"📐 Wie wählt man eps überhaupt?" live beantwortet wird.
"""
)
st.caption(
    "Anders als die Fall-Demos im Portfolio, die an einem Anwendungsfall mehrere Verfahren "
    "vergleichen, zeigt diese Demo - Teil der wachsenden \"Konzepte\"-Reihe - **ein** Verfahren "
    "an einem wachsenden Beispiel: nicht der Zufall der Startpunkte (wie bei k-Means im "
    "vorigen Stück dieser Reihe) ist hier die Schwierigkeitsachse, sondern die Wahl der "
    "beiden DBSCAN-Parameter eps und min_samples."
)

with st.expander("So funktioniert DBSCAN", expanded=True):
    st.markdown(
        """
DBSCAN braucht nur zwei Parameter: **eps** (Suchradius) und **min_samples** (wie viele
Punkte - inklusive des Punktes selbst - mindestens im Radius eps liegen müssen). Damit
klassifiziert es jeden Punkt in genau eine von drei Rollen:

- **Core-Punkt**: mindestens `min_samples` Punkte liegen in seinem eps-Radius - der Kern
  einer dichten Region.
- **Rand-Punkt (Border)**: selbst kein Core-Punkt, aber im eps-Radius eines Core-Punkts -
  gehört zu dessen Cluster, erweitert das Cluster aber nicht weiter.
- **Noise**: weder Core- noch Rand-Punkt - zu weit von jeder dichten Region entfernt, wird
  keinem Cluster zugeordnet.

Ein Cluster entsteht, indem man von einem Core-Punkt aus alle über eine Kette von
Core-Punkten erreichbaren Punkte einsammelt ("dichte-verbunden" - Details im Abschnitt
"📐 Mathematische Formulierung"). Zwei Eigenschaften unterscheiden das grundlegend von
k-Means (dem vorigen Stück dieser Reihe):

- **k muss nicht vorher feststehen** - die Anzahl der Cluster ergibt sich aus den Daten,
  nicht aus einem Regler.
- **Cluster müssen nicht konvex/kugelförmig sein** - DBSCAN folgt der tatsächlichen Form
  einer dichten Region, k-Means könnte z. B. zwei ineinander verschlungene Halbmonde
  grundsätzlich nicht korrekt trennen (ausprobierbar im Preset "Nicht-konvexe Formen").

Die Punktwolke weiter unten zeigt das live: ein Animationsschritt ist ein besuchter Punkt,
Farben markieren die Cluster-Zugehörigkeit, ausgefüllte Punkte sind Core-, umrandete
Punkte Rand-Punkte, graue Kreuze Noise.
        """
    )

st.caption("🎯 Schnellstart – ein Beispielszenario laden:")
PRESET_HELP = {
    "Einfaches Beispiel (gleichmäßige Dichte)": "Mehrere klar getrennte, ähnlich dichte Gruppen - ein eps aus einer weiten Spanne funktioniert.",
    "Nicht-konvexe Formen (zwei Halbmonde)": "Das klassische Zwei-Halbmonde-Beispiel - k-Means könnte diese Form nicht korrekt trennen, DBSCAN schon.",
    "Schwerer Fall (sehr unterschiedliche Dichte)": "Eine dichte, enge Gruppe plus eine lockere, diffuse Gruppe - kein einzelnes eps erfasst beide gleich gut.",
    "Viele Ausreißer (Noise-Erkennung im Fokus)": "Mehrere Gruppen plus deutlich mehr verstreute Einzelpunkte - zeigt, wie DBSCAN Ausreißer explizit als Noise markiert statt sie zu erzwingen.",
}
preset_cols = st.columns(len(C.PRESETS))
for i, name in enumerate(C.PRESETS.keys()):
    with preset_cols[i]:
        st.button(name, width="stretch", on_click=apply_preset, args=(name,), help=PRESET_HELP[name])

st.caption(
    "🔗 Die Adresszeile oben spiegelt Ihre aktuelle Konfiguration wider – einfach kopieren, "
    "um ein Szenario zu teilen."
)

load_permalink_settings()
init_session_state_defaults()

with st.sidebar:
    st.header("⚙️ Einstellungen")
    n_points = st.slider("Anzahl Adressen", *bounds("n_points_slider"), key="n_points_slider")
    k = st.slider(
        "Anzahl Gruppen", *bounds("k_slider"), key="k_slider",
        help="Bei „Gruppen“: Anzahl runder Cluster. Bei „Halbmonde“: Anzahl nicht-konvexer "
        "Bögen (bei 2 die klassische Zwei-Halbmonde-Form, darüber wie Blütenblätter "
        "angeordnet).",
    )
    spread = st.slider(
        "Streuung / Rauschen", *bounds("spread_slider"), key="spread_slider", step=0.05,
        help="Bei „Gruppen“: Streuung je Gruppe. Bei „Halbmonde“: Rauschen um die ideale Kurve.",
    )
    density_imbalance = st.slider(
        "Dichte-Ungleichgewicht", *bounds("density_imbalance_slider"), key="density_imbalance_slider", step=0.05,
        help="0 = alle Gruppen gleich dicht. 1 = eine Gruppe wird deutlich lockerer/diffuser "
        "als die übrigen, bei gleicher Punktzahl - simuliert stark unterschiedliche Dichte. "
        "Wirkt bei beiden Formen.",
    )
    seed = st.number_input("Zufalls-Seed", *bounds("seed_input"), key="seed_input", step=1)

    st.markdown("**Punktwolken-Form**")
    shape = st.radio(
        "Form", options=C.SHAPES, key="shape_radio", format_func=lambda s: C.SHAPE_LABELS[s],
        help="„Gruppen“: runde, konvexe Cluster. „Halbmonde“: nicht-konvexe Bögen - Anzahl "
        "Gruppen und Dichte-Ungleichgewicht wirken auf beide Formen.",
    )
    outlier_fraction = st.slider(
        "Anteil verstreuter Ausreißer", *bounds("outlier_fraction_slider"), key="outlier_fraction_slider", step=0.05,
        help="Zusätzliche, gleichverteilt verstreute Punkte, die zu keiner echten Gruppe "
        "gehören - zeigt, ob DBSCAN sie korrekt als Noise erkennt.",
    )

    st.markdown("**DBSCAN-Parameter**")
    eps = st.slider(
        "eps (Suchradius)", *bounds("eps_slider"), key="eps_slider", step=0.05,
        help="Wie weit DBSCAN nach Nachbarn sucht. Zu klein: fast alles wird Noise. Zu groß: "
        "fast alles verschmilzt zu einem Cluster.",
    )
    min_samples = st.slider(
        "min_samples", *bounds("min_samples_slider"), key="min_samples_slider",
        help="Wie viele Punkte (inklusive des Punktes selbst) mindestens im eps-Radius liegen "
        "müssen, damit ein Punkt Core-Punkt ist.",
    )

    st.button(
        "🎲 Neue Punktwolke generieren",
        width="stretch",
        on_click=randomize_seed,
        help="Würfelt einen neuen Zufalls-Seed für die Adressen.",
    )

sync_query_params(n_points, k, spread, density_imbalance, outlier_fraction, seed, shape, eps, min_samples)

with st.spinner("Führe DBSCAN aus..."):
    instance, result = _compute_run(
        int(n_points), int(k), spread, density_imbalance, outlier_fraction, shape, int(seed), eps, int(min_samples)
    )
    k_distances, sweep = _compute_eps_tools(instance, int(min_samples))

max_step = result.n_events - 1
run_key = (n_points, k, spread, density_imbalance, outlier_fraction, shape, seed, eps, min_samples)
if "db_step" not in st.session_state or st.session_state.get("db_step_owner") != run_key:
    st.session_state["db_step"] = max_step
    st.session_state["db_step_owner"] = run_key

st.markdown("## 🎯 DBSCAN in Aktion")

step_col, play_col = st.columns([5, 1])
with step_col:
    if max_step == 0:
        step = 0
        st.caption("Nur ein einziger Klassifikations-Schritt möglich - kein Regler nötig.")
    else:
        step = st.slider(
            "Schritt (besuchter Punkt)", 0, max_step, key="db_step",
            help="Ein Schritt = ein Punkt wird (erstmals oder erneut, falls zuvor Noise) "
            "klassifiziert.",
        )
with play_col:
    auto_play = st.button("▶️ Abspielen", width="stretch")

scatter_slot = st.empty()


def _render(current_step):
    scatter_slot.plotly_chart(
        build_scatter_figure(instance, result, current_step), width="stretch", key=f"scatter_{current_step}"
    )


if auto_play:
    n_frames = min(max_step + 1, 40)
    frame_skip = max(1, (max_step + 1) // n_frames)
    for s in list(range(0, max_step, frame_skip)) + [max_step]:
        _render(s)
        time.sleep(0.05)
    step = max_step
else:
    _render(step)

labels_now = labels_at_step(instance.n_points, result.events, step)
roles_now = roles_at_step(instance.n_points, result.events, step)
live = stats_at_step(labels_now, roles_now, result, step)

st.caption(f"Aktuelle Parameter: eps={eps:g} · min_samples={int(min_samples)}")

lm1, lm2, lm3, lm4, lm5 = st.columns(5)
lm1.metric("Besuchte Punkte", f"{live['n_visited']}/{instance.n_points}")
lm2.metric("Core-Punkte", live["n_core"])
lm3.metric("Rand-Punkte (Border)", live["n_border"])
lm4.metric("Noise", f"{live['n_noise']} ({live['n_noise'] / instance.n_points * 100:.0f}%)")
lm5.metric("Cluster gefunden", live["n_clusters"])

st.markdown("**Und mit anderen eps-Werten?**")
st.caption(
    "Gleiche Adressen, gleiches min_samples wie oben - nur eps unterscheidet sich, jeweils "
    "als Vielfaches des aktuell eingestellten Werts."
)
example_eps_values = sorted({
    round(min(max(eps * factor, C.EPS_MIN), C.EPS_MAX), 3) for factor in (0.5, 0.75, 1.5, 2.0)
})
example_cols = st.columns(len(example_eps_values))
for col, example_eps in zip(example_cols, example_eps_values):
    with col:
        example_result = _compute_example_run(instance, example_eps, int(min_samples))
        st.plotly_chart(
            build_mini_scatter_figure(instance, example_result), width="stretch", key=f"mini_{example_eps}"
        )
        noise_count = sum(1 for l in example_result.final_labels if l == -1)
        noise_share = noise_count / instance.n_points
        st.caption(
            f"eps={example_eps:g} · {example_result.n_clusters} Cluster · "
            f"{noise_count} ({noise_share * 100:.0f}%) Noise"
        )

st.markdown("---")

st.subheader("📐 Wie wählt man eps überhaupt?")
st.markdown(
    """
Zwei Werkzeuge helfen dabei, kein Zufallstreffer: der **k-Distanz-Plot** (Ester et al.,
1996) zeigt für jeden Punkt die Distanz zu seinem am weitesten entfernten "noch
ausreichenden" Nachbarn, aufsteigend sortiert - ein deutlicher Knick markiert einen guten
eps-Kandidaten. Und live für Ihr aktuelles Szenario geprüft, nicht nur behauptet: wie
unterschiedlich der Noise-Anteil **je wahrer Gruppe** über eine Spanne von eps-Werten
ausfällt.
"""
)

kd_col, sweep_col = st.columns(2)
with kd_col:
    st.plotly_chart(build_k_distance_chart(k_distances, int(min_samples)), width="stretch", key="k_distance")
with sweep_col:
    st.plotly_chart(build_eps_sweep_chart(sweep), width="stretch", key="eps_sweep")

group_fractions = per_group_noise_fraction(instance.true_labels, result.final_labels)
gap = max(group_fractions.values()) - min(group_fractions.values()) if group_fractions else 0.0

gm_cols = st.columns(len(group_fractions) or 1)
for col, (group, fraction) in zip(gm_cols, sorted(group_fractions.items())):
    col.metric(
        f"Noise-Anteil Gruppe {group + 1} (bei aktuellem eps)", f"{fraction * 100:.0f}%",
        help="Anteil der Punkte dieser tatsächlichen Gruppe, die DBSCAN bei den aktuellen "
        "Reglern als Noise einstuft - echte Ausreißerpunkte (falls vorhanden) zählen hier "
        "nicht mit.",
    )

if gap > 0.2:
    worst_group = max(group_fractions, key=group_fractions.get)
    best_group = min(group_fractions, key=group_fractions.get)
    st.success(
        f"✅ Bei diesem Szenario liegt Gruppe {worst_group + 1} bei {group_fractions[worst_group] * 100:.0f}% "
        f"Noise, Gruppe {best_group + 1} dagegen nur bei {group_fractions[best_group] * 100:.0f}% - "
        f"dasselbe eps bedient beide Gruppen sehr unterschiedlich gut, weil sie sich in ihrer "
        f"Dichte deutlich unterscheiden."
    )
else:
    st.info(
        "Bei diesem Szenario ist der Unterschied zwischen den Gruppen noch klein - ein "
        "größeres Dichte-Ungleichgewicht (Regler links) macht ihn deutlicher."
    )

st.markdown("---")

with st.expander("📐 Mathematische Formulierung"):
    st.markdown(
        r"""
**eps-Nachbarschaft** eines Punkts $p$: $N_{\varepsilon}(p) = \{q \in D : \lVert p - q
\rVert \le \varepsilon\}$ (inklusive $p$ selbst).

**Core-Punkt**: $p$ ist Core-Punkt, wenn $|N_{\varepsilon}(p)| \ge \text{minSamples}$.

**Direkt dichte-erreichbar**: $q$ ist von $p$ aus direkt dichte-erreichbar, wenn $q \in
N_{\varepsilon}(p)$ und $p$ Core-Punkt ist. **Dichte-erreichbar**: die transitive Hülle
davon (eine Kette von Core-Punkten). **Dichte-verbunden**: $p$ und $q$ sind
dichte-verbunden, wenn ein Punkt $o$ existiert, von dem aus beide dichte-erreichbar sind.

**Cluster** $C$: eine maximale, nicht-leere Menge dichte-verbundener Punkte. **Noise**:
alle Punkte, die zu keinem Cluster gehören (weder Core- noch Rand-Punkt).

Naive Laufzeit $O(n^2)$ - jede Regionsabfrage $N_{\varepsilon}(p)$ ist linear in $n$, und
es gibt $n$ davon. Mit einer räumlichen Indexstruktur (z. B. einem k-d-Baum) sinkt das in
der Praxis auf $O(n \log n)$ - hier bewusst nicht implementiert, da bei Demo-Größen
($n \le$ {n_max}) der Unterschied nicht spürbar ist.

**k-Distanz-Heuristik** (Ester et al., 1996, "A density-based algorithm for discovering
clusters in large spatial databases with noise"): sortiert man für jeden Punkt die Distanz
zu seinem $(\text{minSamples}-1)$-ten nächsten Nachbarn aufsteigend, markiert ein deutlicher
Knick im resultierenden Diagramm einen eps-Wert, ab dem die meisten Punkte "genug" Nachbarn
haben.

**Und bei noch stärker unterschiedlicher Dichte?** Selbst das beste eps ist hier ein
Kompromiss - siehe die "📐"-Sektion oben. **HDBSCAN** behebt das grundsätzlich, indem es
nicht ein einzelnes eps, sondern eine ganze Hierarchie über alle Dichteschwellen zugleich
betrachtet und für jeden Ast der Hierarchie die stabilste Clusterung extrahiert - deutlich
aufwendiger als DBSCAN, aber genau die Fortsetzung, die sich aus der obigen Beobachtung
ergibt.

Implementiert in `db_algorithm.py` (DBSCAN mit Besuchsprotokoll) und `db_evaluation.py`
(k-Distanz, eps-Sweep).
        """.replace("{n_max}", str(C.N_POINTS_MAX))
    )

st.markdown("---")

st.caption(
    "Diese Demo ist Teil des Portfolios von [Sebastian Hanisch](https://sebastianhanisch.net) – "
    "Operations Research und Machine Learning. Interesse an einer maßgeschneiderten Lösung für "
    "Ihr Unternehmen? [Kontakt aufnehmen](https://sebastianhanisch.net/kontakt.html)"
)
