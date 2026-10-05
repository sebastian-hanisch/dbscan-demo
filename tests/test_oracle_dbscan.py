"""Orakel-Test: DBSCAN gegen einen unabhängigen Rechenweg (Distanzmatrix + Zusammenhangskomponenten
der Core-Punkte) und gegen sklearn.cluster.DBSCAN, auch auf Ganzzahl-Gittern mit Gleichständen
und Duplikaten. Randpunkte hängen von der Besuchsreihenfolge ab und werden nur auf Gültigkeit
geprüft (Cluster mit einem Core-Punkt in Reichweite), nicht exakt verglichen."""

import numpy as np
import pytest

from db_algorithm import labels_at_step, run


def _reference(data, eps, min_samples):
    n = len(data)
    dist = np.sqrt(((data[:, None, :] - data[None, :, :]) ** 2).sum(-1))
    adj = dist <= eps
    core = adj.sum(1) >= min_samples
    comp = -np.ones(n, dtype=int)
    n_comp = 0
    for s in range(n):
        if core[s] and comp[s] < 0:
            stack = [s]
            comp[s] = n_comp
            while stack:
                u = stack.pop()
                for v in np.nonzero(adj[u] & core & (comp < 0))[0]:
                    comp[v] = n_comp
                    stack.append(v)
            n_comp += 1
    return core, comp, adj, n_comp


def test_dbscan_matches_independent_reference_and_sklearn():
    sk_cluster = pytest.importorskip("sklearn.cluster")
    rng = np.random.default_rng(0)
    for trial in range(120):
        if trial % 2 == 0:
            data = rng.integers(0, 6, size=(int(rng.integers(5, 40)), 2)).astype(float)
            eps = float(rng.choice([1.0, 2.0, 3.0]))
        else:
            data = rng.normal(size=(int(rng.integers(5, 60)), 2))
            eps = float(rng.uniform(0.1, 1.5))
        min_samples = int(rng.integers(2, 8))
        result = run(data, eps, min_samples)
        core, comp, adj, n_comp = _reference(data, eps, min_samples)
        labels = np.array(result.final_labels)

        assert {i for i, r in enumerate(result.final_roles) if r == "core"} == set(np.nonzero(core)[0])
        assert result.n_clusters == n_comp
        # Core-Partition bis auf Umbenennung gleich
        for a in np.nonzero(core)[0]:
            for b in np.nonzero(core)[0]:
                assert (labels[a] == labels[b]) == (comp[a] == comp[b])
        for i in np.nonzero(~core)[0]:
            near_core = np.nonzero(adj[i] & core)[0]
            if len(near_core) == 0:
                assert labels[i] == -1 and result.final_roles[i] == "noise"
            else:
                assert result.final_roles[i] == "border"
                assert labels[i] in {labels[j] for j in near_core}

        sk = sk_cluster.DBSCAN(eps=eps, min_samples=min_samples).fit(data)
        assert set(sk.core_sample_indices_.tolist()) == set(np.nonzero(core)[0].tolist())
        assert set(np.nonzero(sk.labels_ == -1)[0]) == set(np.nonzero(labels == -1)[0])

        rebuilt = labels_at_step(len(data), result.events, result.n_events - 1)
        assert tuple(int(x) for x in rebuilt) == result.final_labels
        assert result.n_events <= 2 * len(data)
