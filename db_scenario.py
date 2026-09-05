"""Zufällige 2D-Punktwolken für die DBSCAN-Demo: entweder k Gauß-Cluster mit
einstellbarem Dichte-Ungleichgewicht ("blobs") oder k nicht-konvexe Halbkreis-Bögen
("moons") - bei k=2 exakt das klassische Zwei-Halbmonde-Beispiel, bei k>2 k Bögen wie
Blütenblätter auf einem Ring angeordnet, jeweils mit konkaver Seite zum Zentrum - plus
optional gleichmäßig verstreute Ausreißerpunkte obendrauf, die keiner echten Gruppe
angehören (true_label -1). k und density_imbalance wirken bei BEIDEN Formen."""

from dataclasses import dataclass

import numpy as np

RING_RADIUS = 3.0
ARC_RADIUS = 2.5
ARC_RING_RADIUS = 6.5
MIN_STD_FRACTION = 0.05


@dataclass(frozen=True)
class ClusteringInstance:
    points: tuple  # ((x, y), ...)
    true_labels: tuple  # Gruppenindex, oder -1 für verstreute Ausreißer
    shape: str  # "blobs" oder "moons"
    k: int  # Gruppenanzahl, bei BEIDEN Formen bedeutungsvoll

    @property
    def n_points(self):
        return len(self.points)

    def as_array(self):
        return np.array(self.points, dtype=float)


def _group_scales(k, spread, density_imbalance, base):
    """Gruppe 0 wird mit wachsendem density_imbalance diffuser (groessere Streuung bei
    GLEICHER Punktzahl, also geringere lokale Dichte), die uebrigen Gruppen entsprechend
    dichter - anders als kmeans-demos Groessen-Ungleichgewicht bleibt hier die Punktzahl
    je Gruppe konstant, nur die Dichte variiert. `base` ist die Basis-Streuung/das
    Basis-Rauschen vor der Ungleichgewichts-Skalierung (Blobs und Boegen nutzen dieselbe
    Funktion mit unterschiedlichem `base`)."""
    scales = np.full(k, base)
    if k > 1:
        scales[0] *= 1 + density_imbalance
        scales[1:] *= max(1 - 0.6 * density_imbalance, MIN_STD_FRACTION)
    return scales


def _generate_blobs(n_points, k, spread, density_imbalance, rng):
    angles = np.linspace(0, 2 * np.pi, k, endpoint=False) + rng.uniform(-0.15, 0.15, size=k)
    centers = np.stack([RING_RADIUS * np.cos(angles), RING_RADIUS * np.sin(angles)], axis=1)
    base_std = max(spread, MIN_STD_FRACTION) * RING_RADIUS
    stds = _group_scales(k, spread, density_imbalance, base_std)

    counts = np.full(k, n_points // k)
    counts[-1] += n_points - counts.sum()

    points_per_cluster, labels_per_cluster = [], []
    for i in range(k):
        pts = rng.normal(loc=centers[i], scale=stds[i], size=(counts[i], 2))
        points_per_cluster.append(pts)
        labels_per_cluster.append(np.full(counts[i], i))
    return np.concatenate(points_per_cluster, axis=0), np.concatenate(labels_per_cluster, axis=0)


def _generate_moons(n_points, k, spread, density_imbalance, rng):
    """k=2: das klassische "two moons"-Beispiel (wie sklearn.datasets.make_moons, hier
    from scratch nachgebaut) - zwei ineinander verschlungene Halbkreise, unveraendert
    gegenueber der urspruenglichen Fassung dieser Demo, damit das bekannte Bild erhalten
    bleibt. k>2: k Halbkreis-Boegen wie Bluetenblaetter auf einem Ring angeordnet, jeder
    mit seiner konkaven Seite zum Ringzentrum - weiterhin klar nicht-konvex je Gruppe
    (der Punkt der Halbmond-Form), nur mit variabler Gruppenzahl. In beiden Faellen wirkt
    density_imbalance wie bei Blobs: Gruppe 0 wird diffuser, die uebrigen enger."""
    counts = np.full(k, n_points // k)
    counts[-1] += n_points - counts.sum()
    base_noise_std = max(spread, MIN_STD_FRACTION) * ARC_RADIUS * 0.3
    noise_stds = _group_scales(k, spread, density_imbalance, base_noise_std)

    if k == 2:
        t1 = rng.uniform(0, np.pi, counts[0])
        x1 = ARC_RADIUS * np.cos(t1)
        y1 = ARC_RADIUS * np.sin(t1)

        t2 = rng.uniform(0, np.pi, counts[1])
        x2 = ARC_RADIUS * (1 - np.cos(t2))
        y2 = ARC_RADIUS * (0.5 - np.sin(t2))

        pts1 = np.stack([x1, y1], axis=1) + rng.normal(scale=noise_stds[0], size=(counts[0], 2))
        pts2 = np.stack([x2, y2], axis=1) + rng.normal(scale=noise_stds[1], size=(counts[1], 2))
        points = np.concatenate([pts1, pts2], axis=0)
        labels = np.concatenate([np.zeros(counts[0], dtype=int), np.ones(counts[1], dtype=int)])
        return points, labels

    layout_angles = np.linspace(0, 2 * np.pi, k, endpoint=False) + rng.uniform(-0.1, 0.1, size=k)
    arc_centers = np.stack(
        [ARC_RING_RADIUS * np.cos(layout_angles), ARC_RING_RADIUS * np.sin(layout_angles)], axis=1
    )

    points_per_group, labels_per_group = [], []
    for i in range(k):
        t = rng.uniform(0, np.pi, counts[i])
        local_x = ARC_RADIUS * np.cos(t)
        local_y = ARC_RADIUS * np.sin(t)
        # Um layout_angle_i + pi rotieren, damit die konkave Seite des Bogens zum
        # Ringzentrum zeigt (Bluetenblatt-Anordnung), statt nach aussen.
        rot = layout_angles[i] + np.pi
        cos_r, sin_r = np.cos(rot), np.sin(rot)
        rx = cos_r * local_x - sin_r * local_y
        ry = sin_r * local_x + cos_r * local_y
        pts = np.stack([rx, ry], axis=1) + arc_centers[i] + rng.normal(scale=noise_stds[i], size=(counts[i], 2))
        points_per_group.append(pts)
        labels_per_group.append(np.full(counts[i], i))

    return np.concatenate(points_per_group, axis=0), np.concatenate(labels_per_group, axis=0)


def generate_instance(n_points, k, spread, density_imbalance, outlier_fraction, shape, seed):
    """spread steuert je nach shape entweder die Cluster-Streuung ("blobs") oder das
    Rauschen um die ideale Bogen-Kurve ("moons"). k und density_imbalance wirken bei
    BEIDEN Formen. outlier_fraction erzeugt zusaetzliche, gleichverteilt verstreute
    Punkte (true_label -1), die zu keiner echten Gruppe gehoeren - das Vehikel fuer die
    "erkennt DBSCAN Ausreisser?"-Frage."""
    rng = np.random.default_rng(seed)

    if shape == "moons":
        points, labels = _generate_moons(n_points, k, spread, density_imbalance, rng)
    else:
        points, labels = _generate_blobs(n_points, k, spread, density_imbalance, rng)

    n_outliers = int(round(outlier_fraction * n_points))
    if n_outliers > 0:
        margin = 1.5
        xmin, xmax = points[:, 0].min() - margin, points[:, 0].max() + margin
        ymin, ymax = points[:, 1].min() - margin, points[:, 1].max() + margin
        outlier_points = np.stack(
            [rng.uniform(xmin, xmax, n_outliers), rng.uniform(ymin, ymax, n_outliers)], axis=1
        )
        points = np.concatenate([points, outlier_points], axis=0)
        labels = np.concatenate([labels, np.full(n_outliers, -1)])

    return ClusteringInstance(
        points=tuple(map(tuple, points.tolist())),
        true_labels=tuple(int(l) for l in labels),
        shape=shape,
        k=k,
    )
