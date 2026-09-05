import numpy as np

from db_scenario import generate_instance


def test_point_count_matches_request_including_outliers():
    instance = generate_instance(
        n_points=100, k=3, spread=0.2, density_imbalance=0.0, outlier_fraction=0.1, shape="blobs", seed=1
    )
    assert instance.n_points == 110  # 100 Basis-Punkte + 10 (10%) Ausreisser obendrauf
    assert instance.true_labels.count(-1) == 10


def test_reproducible_given_same_seed():
    kwargs = dict(n_points=80, k=3, spread=0.25, density_imbalance=0.3, outlier_fraction=0.05, shape="blobs", seed=42)
    a = generate_instance(**kwargs)
    b = generate_instance(**kwargs)
    assert a.points == b.points
    assert a.true_labels == b.true_labels


def test_no_outliers_by_default_fraction_zero():
    instance = generate_instance(
        n_points=60, k=2, spread=0.2, density_imbalance=0.0, outlier_fraction=0.0, shape="blobs", seed=1
    )
    assert -1 not in instance.true_labels
    assert instance.n_points == 60


def test_density_imbalance_lowers_local_density_of_group_zero():
    """Gruppe 0 soll bei hohem density_imbalance eine deutlich groessere mittlere
    Naechste-Nachbar-Distanz haben als die uebrigen Gruppen (gleiche Punktzahl,
    aber ueber eine groessere Flaeche gestreut - also lokal weniger dicht),
    obwohl alle Gruppen exakt gleich viele Punkte enthalten."""
    instance = generate_instance(
        n_points=150, k=3, spread=0.2, density_imbalance=0.9, outlier_fraction=0.0, shape="blobs", seed=3
    )
    points = np.array(instance.points)
    labels = np.array(instance.true_labels)

    def mean_nn_distance(group_points):
        d = np.sqrt(((group_points[:, None, :] - group_points[None, :, :]) ** 2).sum(axis=2))
        np.fill_diagonal(d, np.inf)
        return d.min(axis=1).mean()

    nn_group0 = mean_nn_distance(points[labels == 0])
    nn_others = np.mean([mean_nn_distance(points[labels == i]) for i in (1, 2)])
    assert nn_group0 > nn_others * 1.5


def test_moons_shape_produces_two_balanced_groups():
    instance = generate_instance(
        n_points=100, k=3, spread=0.08, density_imbalance=0.0, outlier_fraction=0.0, shape="moons", seed=5
    )
    labels = np.array(instance.true_labels)
    assert set(labels.tolist()) == {0, 1}
    assert abs(int((labels == 0).sum()) - int((labels == 1).sum())) <= 1
