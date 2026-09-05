"""Live-Kennzahlen pro Schritt sowie die "Wie wählt man eps überhaupt?"-Werkzeuge:
k-Distanz-Plot (Ester et al., 1996) und eps-Sweep (Cluster-Anzahl/Noise-Anteil über
eine Spanne von eps-Werten) - das DBSCAN-Äquivalent zu kmeans-demos Multistart-Vergleich."""

from dataclasses import dataclass

import numpy as np

from db_algorithm import NOISE, run


def roles_at_step(n_points, events, step):
    roles = [None] * n_points
    for event in events[: step + 1]:
        roles[event.point_index] = event.role
    return roles


def stats_at_step(labels, roles, result, step):
    """`step` zaehlt Ereignisse (ein Punkt, der zuerst als Noise und spaeter als Rand-
    Punkt reklassifiziert wird, erzeugt zwei Ereignisse) - fuer "besuchte Punkte" zaehlt
    stattdessen die Anzahl DISTINKTER Punktindizes unter diesen Ereignissen, damit die
    Anzeige nie ueber die Gesamtpunktzahl hinausgeht."""
    n_clusters_so_far = len({l for l in labels if l >= 0})
    n_distinct_visited = len({event.point_index for event in result.events[: step + 1]})
    return {
        "n_visited": n_distinct_visited,
        "n_core": roles.count("core"),
        "n_border": roles.count("border"),
        "n_noise": roles.count("noise"),
        "n_clusters": n_clusters_so_far,
        "done": step == result.n_events - 1,
    }


def k_distance_values(data, min_samples):
    """Fuer jeden Punkt die Distanz zu seinem (min_samples - 1)-ten naechsten ANDEREN
    Punkt - genau die Anzahl "echter" Nachbarn (ohne sich selbst), die ein Punkt
    braucht, um Core-Punkt zu sein (siehe db_algorithm._region_query, das sich selbst
    mitzaehlt). Aufsteigend sortiert zurueckgegeben, wie im k-Distanz-Diagramm ueblich -
    der "Knick" markiert einen sinnvollen eps-Kandidaten."""
    n = len(data)
    k = max(1, min_samples - 1)
    values = []
    for i in range(n):
        dists = np.sort(np.sqrt(((data - data[i]) ** 2).sum(axis=1)))
        values.append(dists[k] if k < len(dists) else dists[-1])
    return tuple(sorted(values))


def suggested_eps_range(data, min_samples, n_steps):
    """Leitet eine sinnvolle eps-Spanne fuer den Sweep direkt aus den Daten ab (5.-95.
    Perzentil der k-Distanzen, mit Sicherheitsabstand nach beiden Seiten), statt eine
    feste Spanne zu raten, die fuer sehr unterschiedlich skalierte Szenarien nicht passt."""
    kdist = np.array(k_distance_values(data, min_samples))
    lo_idx, hi_idx = int(len(kdist) * 0.05), int(len(kdist) * 0.95)
    lo = max(kdist[lo_idx] * 0.5, 1e-3)
    hi = kdist[hi_idx] * 2.0
    if hi <= lo:
        hi = lo * 2
    return tuple(np.linspace(lo, hi, n_steps))


def per_group_noise_fraction(true_labels, final_labels):
    """Anteil Noise je WAHRER Gruppe (echte Ausreisserpunkte mit true_label -1
    ausgenommen, die sollen ohnehin Noise sein) - macht sichtbar, ob ein eps alle
    Gruppen gleich gut bedient oder nur die dichteste."""
    true_labels = np.asarray(true_labels)
    final_labels = np.asarray(final_labels)
    groups = sorted(set(true_labels.tolist()) - {-1})
    return {g: float(np.mean(final_labels[true_labels == g] == NOISE)) for g in groups}


@dataclass(frozen=True)
class EpsSweepResult:
    eps_values: tuple
    n_clusters: tuple
    noise_fraction: tuple
    per_group_noise_fraction: dict  # {Gruppenindex: tuple, gleiche Laenge wie eps_values}


def eps_sweep(data, min_samples, eps_values, true_labels=None):
    n = len(data)
    n_clusters_list, noise_fraction_list = [], []
    per_group_lists = {}
    for eps in eps_values:
        result = run(data, eps, min_samples)
        n_clusters_list.append(result.n_clusters)
        noise_fraction_list.append(sum(1 for l in result.final_labels if l == NOISE) / n)
        if true_labels is not None:
            for group, fraction in per_group_noise_fraction(true_labels, result.final_labels).items():
                per_group_lists.setdefault(group, []).append(fraction)
    return EpsSweepResult(
        eps_values=tuple(eps_values),
        n_clusters=tuple(n_clusters_list),
        noise_fraction=tuple(noise_fraction_list),
        per_group_noise_fraction={g: tuple(vals) for g, vals in per_group_lists.items()},
    )
