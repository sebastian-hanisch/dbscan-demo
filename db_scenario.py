"""Zufällige 2D-Punktwolken für die DBSCAN-Demo: entweder k Gauß-Cluster mit
einstellbarem Dichte-Ungleichgewicht ("blobs") oder das klassische, nicht-konvexe
Zwei-Halbmonde-Beispiel ("moons") - plus optional gleichmäßig verstreute
Ausreißerpunkte obendrauf, die keiner echten Gruppe angehören (true_label -1)."""

from dataclasses import dataclass

import numpy as np

RING_RADIUS = 3.0
MOON_RADIUS = 2.5
MIN_STD_FRACTION = 0.05


@dataclass(frozen=True)
class ClusteringInstance:
    points: tuple  # ((x, y), ...)
    true_labels: tuple  # Gruppenindex, oder -1 für verstreute Ausreißer
    shape: str  # "blobs" oder "moons"
    k: int  # nur bei shape="blobs" bedeutungsvoll

    @property
    def n_points(self):
        return len(self.points)

    def as_array(self):
        return np.array(self.points, dtype=float)


def _blob_stds(k, spread, density_imbalance):
    """Cluster 0 wird mit wachsendem density_imbalance diffuser (groessere Streuung
    bei GLEICHER Punktzahl, also geringere lokale Dichte), die uebrigen Cluster
    entsprechend dichter - anders als kmeans-demos Groessen-Ungleichgewicht bleibt
    hier die Punktzahl je Gruppe konstant, nur die Dichte variiert."""
    base_std = max(spread, MIN_STD_FRACTION) * RING_RADIUS
    stds = np.full(k, base_std)
    if k > 1:
        stds[0] *= 1 + density_imbalance
        stds[1:] *= max(1 - 0.6 * density_imbalance, MIN_STD_FRACTION)
    return stds


def _generate_blobs(n_points, k, spread, density_imbalance, rng):
    angles = np.linspace(0, 2 * np.pi, k, endpoint=False) + rng.uniform(-0.15, 0.15, size=k)
    centers = np.stack([RING_RADIUS * np.cos(angles), RING_RADIUS * np.sin(angles)], axis=1)
    stds = _blob_stds(k, spread, density_imbalance)

    counts = np.full(k, n_points // k)
    counts[-1] += n_points - counts.sum()

    points_per_cluster, labels_per_cluster = [], []
    for i in range(k):
        pts = rng.normal(loc=centers[i], scale=stds[i], size=(counts[i], 2))
        points_per_cluster.append(pts)
        labels_per_cluster.append(np.full(counts[i], i))
    return np.concatenate(points_per_cluster, axis=0), np.concatenate(labels_per_cluster, axis=0)


def _generate_moons(n_points, spread, rng):
    """Klassisches "two moons"-Beispiel (wie sklearn.datasets.make_moons, hier from
    scratch nachgebaut): zwei ineinander verschlungene Halbkreise - nicht-konvex,
    k-Means koennte diese Form grundsaetzlich nicht korrekt trennen, DBSCAN schon."""
    n1 = n_points // 2
    n2 = n_points - n1

    t1 = rng.uniform(0, np.pi, n1)
    x1 = MOON_RADIUS * np.cos(t1)
    y1 = MOON_RADIUS * np.sin(t1)

    t2 = rng.uniform(0, np.pi, n2)
    x2 = MOON_RADIUS * (1 - np.cos(t2))
    y2 = MOON_RADIUS * (0.5 - np.sin(t2))

    noise_std = max(spread, MIN_STD_FRACTION) * MOON_RADIUS * 0.3
    pts1 = np.stack([x1, y1], axis=1) + rng.normal(scale=noise_std, size=(n1, 2))
    pts2 = np.stack([x2, y2], axis=1) + rng.normal(scale=noise_std, size=(n2, 2))

    points = np.concatenate([pts1, pts2], axis=0)
    labels = np.concatenate([np.zeros(n1, dtype=int), np.ones(n2, dtype=int)])
    return points, labels


def generate_instance(n_points, k, spread, density_imbalance, outlier_fraction, shape, seed):
    """spread steuert je nach shape entweder die Cluster-Streuung ("blobs") oder das
    Rauschen um die ideale Halbmond-Kurve ("moons"). outlier_fraction erzeugt
    zusaetzliche, gleichverteilt verstreute Punkte (true_label -1), die zu keiner
    echten Gruppe gehoeren - das Vehikel fuer die "erkennt DBSCAN Ausreisser?"-Frage."""
    rng = np.random.default_rng(seed)

    if shape == "moons":
        points, labels = _generate_moons(n_points, spread, rng)
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
