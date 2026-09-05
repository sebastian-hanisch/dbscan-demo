import numpy as np
import pytest

from db_algorithm import NOISE, labels_at_step, run
from db_scenario import generate_instance


def test_hand_computed_line_of_points():
    """Vier gleichmaessig verteilte Punkte auf einer Linie (Abstand 1) plus ein
    weit entfernter Ausreisser, eps=1.5, min_samples=3: von Hand nachvollziehbar,
    welcher Punkt core/border/noise ist, unabhaengig von der Besuchsreihenfolge -
    p1 und p2 haben je 3 Nachbarn (inkl. sich selbst) im eps-Radius und sind damit
    core, p0/p3 haben nur 2 und sind border (an p1 bzw. p2 angebunden), p4 ist zu
    weit weg von jedem core-Punkt und bleibt noise."""
    data = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [3.0, 0.0], [10.0, 0.0]])
    result = run(data, eps=1.5, min_samples=3)

    assert result.final_roles == ("border", "core", "core", "border", "noise")
    assert result.final_labels[:4] == (0, 0, 0, 0)
    assert result.final_labels[4] == NOISE
    assert result.n_clusters == 1


def test_far_outlier_alone_is_always_noise():
    data = np.array([[0.0, 0.0], [0.1, 0.0], [0.0, 0.1], [50.0, 50.0]])
    result = run(data, eps=1.0, min_samples=3)
    assert result.final_roles[3] == "noise"
    assert result.final_labels[3] == NOISE


def test_every_point_classified_exactly_once_per_role():
    instance = generate_instance(n_points=120, k=3, spread=0.3, density_imbalance=0.4, outlier_fraction=0.1, shape="blobs", seed=5)
    result = run(instance.as_array(), eps=0.6, min_samples=5)
    assert len(result.final_roles) == instance.n_points
    assert set(result.final_roles) <= {"core", "border", "noise"}
    non_noise_clusters = {label for label, role in zip(result.final_labels, result.final_roles) if role != "noise"}
    assert non_noise_clusters == set(range(result.n_clusters))


def test_labels_at_step_matches_final_state_at_last_step():
    instance = generate_instance(n_points=80, k=2, spread=0.25, density_imbalance=0.0, outlier_fraction=0.05, shape="blobs", seed=6)
    result = run(instance.as_array(), eps=0.6, min_samples=5)
    reconstructed = labels_at_step(instance.n_points, result.events, result.n_events - 1)
    assert tuple(int(l) for l in reconstructed) == result.final_labels


def _partitions_match(labels_a, labels_b, noise_value_a, noise_value_b):
    """Vergleicht zwei Label-Zuweisungen bis auf Umbenennung der Cluster-IDs:
    Noise muss an denselben Positionen liegen, und zwei Nicht-Noise-Punkte
    muessen unter a genau dann im selben Cluster sein, wenn unter b."""
    noise_a = np.array(labels_a) == noise_value_a
    noise_b = np.array(labels_b) == noise_value_b
    if not np.array_equal(noise_a, noise_b):
        return False
    non_noise_idx = np.nonzero(~noise_a)[0]
    for i in range(len(non_noise_idx)):
        for j in range(i + 1, len(non_noise_idx)):
            p, q = non_noise_idx[i], non_noise_idx[j]
            same_a = labels_a[p] == labels_a[q]
            same_b = labels_b[p] == labels_b[q]
            if same_a != same_b:
                return False
    return True


def test_matches_sklearn_dbscan_partition_and_core_points():
    """Unabhaengiger Kreuzvergleich der eigenen Implementierung gegen
    sklearn.cluster.DBSCAN: dieselbe Partition (bis auf Cluster-ID-Umbenennung),
    dieselben Noise-Punkte, dieselbe Menge an Core-Punkten. sklearn ist
    ausschliesslich ein Test-Dependency, kein Laufzeit-Dependency der App."""
    sklearn = pytest.importorskip("sklearn.cluster")

    scenarios = [
        dict(n_points=100, k=3, spread=0.25, density_imbalance=0.0, outlier_fraction=0.1, shape="blobs", seed=1),
        dict(n_points=120, k=2, spread=0.15, density_imbalance=0.8, outlier_fraction=0.0, shape="blobs", seed=2),
        dict(n_points=90, k=3, spread=0.08, density_imbalance=0.0, outlier_fraction=0.0, shape="moons", seed=3),
    ]
    param_combos = [(0.4, 5), (0.6, 4), (0.3, 6)]

    for scenario_kwargs in scenarios:
        instance = generate_instance(**scenario_kwargs)
        data = instance.as_array()
        for eps, min_samples in param_combos:
            ours = run(data, eps=eps, min_samples=min_samples)
            sk_model = sklearn.DBSCAN(eps=eps, min_samples=min_samples).fit(data)

            assert _partitions_match(ours.final_labels, tuple(sk_model.labels_), NOISE, -1)

            our_core = {i for i, role in enumerate(ours.final_roles) if role == "core"}
            sk_core = set(sk_model.core_sample_indices_.tolist())
            assert our_core == sk_core
