"""DBSCAN from scratch, mit vollständigem Besuchsprotokoll, damit die App Punkt für
Punkt (nicht nur das Endergebnis) durchblättern kann - analog zum Knoten-Protokoll von
Branch & Bound (branch-bound-demo/bb_solver.py) und dem Iterations-Protokoll von
Lloyd's Algorithmus (kmeans-demo/km_algorithm.py).

Bewusst ohne sklearn zur Laufzeit implementiert. sklearn.cluster.DBSCAN dient in
tests/ nur als unabhängiger Kreuzvergleich für die eigene Implementierung.
"""

from collections import deque
from dataclasses import dataclass

import numpy as np

NOISE = -1
UNVISITED = -2


@dataclass(frozen=True)
class Event:
    step: int  # Reihenfolge, in der Punkte (erst-)klassifiziert oder umklassifiziert wurden
    point_index: int
    role: str  # "core" | "border" | "noise"
    cluster: int  # Cluster-ID, oder NOISE (-1)


@dataclass(frozen=True)
class RunResult:
    events: tuple  # Event-Folge in chronologischer Reihenfolge, ein Event je (Um-)Klassifizierung
    final_labels: tuple  # Cluster-ID je Punkt, NOISE (-1) fuer Ausreisser
    final_roles: tuple  # "core" | "border" | "noise" je Punkt, finaler Stand
    n_clusters: int

    @property
    def n_events(self):
        return len(self.events)


def _region_query(data, point_index, eps):
    """Alle Punkte (inklusive des Punktes selbst) innerhalb von eps - Selbsteinschluss
    passend zu sklearn.cluster.DBSCANs min_samples-Konvention (der Kreuzvergleich in
    tests/test_algorithm.py haengt davon ab)."""
    dists = np.sqrt(((data - data[point_index]) ** 2).sum(axis=1))
    return np.nonzero(dists <= eps)[0]


def run(data, eps, min_samples):
    n = len(data)
    labels = np.full(n, UNVISITED)
    roles = [""] * n
    visited = np.zeros(n, dtype=bool)
    events = []
    cluster_id = 0

    def record(idx, role, cluster):
        roles[idx] = role
        labels[idx] = cluster if role != "noise" else NOISE
        events.append(Event(len(events), idx, role, cluster))

    for i in range(n):
        if visited[i]:
            continue
        visited[i] = True
        neighbors = _region_query(data, i, eps)

        if len(neighbors) < min_samples:
            record(i, "noise", NOISE)
            continue

        record(i, "core", cluster_id)
        seeds = deque(neighbors)
        while seeds:
            j = seeds.popleft()
            if not visited[j]:
                visited[j] = True
                j_neighbors = _region_query(data, j, eps)
                if len(j_neighbors) >= min_samples:
                    record(j, "core", cluster_id)
                    seeds.extend(j_neighbors)
                else:
                    record(j, "border", cluster_id)
            elif labels[j] == NOISE:
                record(j, "border", cluster_id)
        cluster_id += 1

    return RunResult(
        events=tuple(events),
        final_labels=tuple(int(l) for l in labels),
        final_roles=tuple(roles),
        n_clusters=cluster_id,
    )


def labels_at_step(n_points, events, step):
    """Rekonstruiert den Label-Zustand nach genau `step` Ereignissen (0-indiziert,
    inklusive) - UNVISITED fuer noch nicht besuchte Punkte, NOISE fuer Ausreisser,
    sonst die Cluster-ID. Treibt die Schritt-fuer-Schritt-Animation der App."""
    labels = np.full(n_points, UNVISITED)
    for event in events[: step + 1]:
        labels[event.point_index] = event.cluster if event.role != "noise" else NOISE
    return labels
