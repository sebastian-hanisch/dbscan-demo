import numpy as np

from db_algorithm import run
from db_evaluation import eps_sweep, k_distance_values, per_group_noise_fraction, suggested_eps_range
from db_scenario import generate_instance


def test_k_distance_values_hand_computed():
    data = np.array([[0.0, 0.0], [1.0, 0.0], [3.0, 0.0], [6.0, 0.0]])

    assert k_distance_values(data, min_samples=2) == (1.0, 1.0, 2.0, 3.0)
    assert k_distance_values(data, min_samples=3) == (2.0, 3.0, 3.0, 5.0)


def test_eps_sweep_extremes_are_almost_all_noise_or_a_single_cluster():
    instance = generate_instance(
        n_points=120, k=3, spread=0.2, density_imbalance=0.0, outlier_fraction=0.0, shape="blobs", seed=1
    )
    data = instance.as_array()

    tiny = eps_sweep(data, min_samples=5, eps_values=[1e-6])
    assert tiny.noise_fraction[0] > 0.95

    huge_eps = float(np.sqrt(((data.max(axis=0) - data.min(axis=0)) ** 2).sum())) * 10
    huge = eps_sweep(data, min_samples=5, eps_values=[huge_eps])
    assert huge.n_clusters[0] == 1
    assert huge.noise_fraction[0] == 0.0


def test_dense_and_sparse_group_need_different_eps_to_avoid_noise():
    """Kern-Behauptung der Demo: bei stark unterschiedlicher Dichte gibt es kein
    eps, das beide Gruppen gleich gut bedient - bei einem eps, das die dichte
    Gruppe schon vollstaendig (0% Noise) erfasst, bleibt die diffuse Gruppe noch
    grossteils Noise."""
    instance = generate_instance(
        n_points=150, k=2, spread=0.12, density_imbalance=0.9, outlier_fraction=0.0, shape="blobs", seed=3
    )
    data = instance.as_array()
    result = run(data, eps=0.3, min_samples=5)
    fractions = per_group_noise_fraction(instance.true_labels, result.final_labels)

    assert fractions[1] == 0.0  # dichte Gruppe: komplett erfasst
    assert fractions[0] > 0.3  # diffuse Gruppe: noch grossteils Noise


def test_eps_sweep_tracks_per_group_noise_when_true_labels_given():
    instance = generate_instance(
        n_points=100, k=2, spread=0.15, density_imbalance=0.7, outlier_fraction=0.0, shape="blobs", seed=4
    )
    sweep = eps_sweep(instance.as_array(), min_samples=5, eps_values=[0.3, 0.6, 0.9], true_labels=instance.true_labels)
    assert set(sweep.per_group_noise_fraction.keys()) == {0, 1}
    assert all(len(vals) == 3 for vals in sweep.per_group_noise_fraction.values())


def test_suggested_eps_range_is_increasing_and_positive():
    instance = generate_instance(
        n_points=100, k=2, spread=0.3, density_imbalance=0.5, outlier_fraction=0.1, shape="blobs", seed=2
    )
    eps_values = suggested_eps_range(instance.as_array(), min_samples=5, n_steps=20)
    assert len(eps_values) == 20
    assert eps_values[0] > 0
    assert all(b > a for a, b in zip(eps_values, eps_values[1:]))
