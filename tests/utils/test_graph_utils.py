import numpy as np
import pytest
from scipy.spatial import KDTree

from torchmil.utils import degree, add_self_loops, normalize_adj, build_adj

def test_degree():
    edge_index = np.array([[0, 1, 2], [1, 2, 0]])  # 0->1, 1->2, 2->0
    edge_weight = np.array([1, 2, 3])
    deg = degree(edge_index, edge_weight, n_nodes=3)
    assert np.array_equal(deg, np.array([1, 2, 3]))

def test_add_self_loops():
    edge_index = np.array([[0, 1], [1, 2]])
    edge_weight = np.array([1.0, 2.0])
    new_edge_index, new_edge_weight = add_self_loops(edge_index, edge_weight, n_nodes=3)

    # Check that self-loops were added to 3 nodes
    assert new_edge_index.shape[1] == 5  # original 2 + 3 self-loops
    assert new_edge_weight.shape[0] == 5

    # Check values
    assert np.allclose(new_edge_weight[-3:], np.ones(3))

def test_normalize_adj():
    edge_index = np.array([[0, 1, 2], [1, 2, 0]])
    edge_weight = np.array([1.0, 1.0, 1.0])
    norm_weights = normalize_adj(edge_index, edge_weight, n_nodes=3)
    assert norm_weights.shape == (3,)
    assert np.all(norm_weights <= 1.0)

def test_build_adj_binary():
    coords = np.array([[0, 0], [0, 1], [1, 0]])
    edge_index, edge_weight = build_adj(coords, feat=None, dist_thr=1.5, add_self_loops=False)

    assert edge_index.shape[0] == 2
    assert edge_index.shape[1] > 0
    assert np.all(edge_weight == 1.0)

def test_build_adj_with_feats():
    coords = np.array([[0, 0], [0, 1], [1, 0]])
    feat = np.array([[1, 0], [0, 1], [1, 1]])
    edge_index, edge_weight = build_adj(coords, feat=feat, dist_thr=1.5, add_self_loops=True)

    assert edge_index.shape[0] == 2
    assert edge_weight.shape[0] == edge_index.shape[1]
    assert np.all(edge_weight >= 0.0)
