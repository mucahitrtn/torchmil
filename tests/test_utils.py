import pytest
import numpy as np

from torchmil.utils import degree, add_self_loops, normalize_adj, build_adj

@pytest.fixture
def sample_graph():
    """
    Pytest fixture to provide a sample graph for testing.
    """
    edge_index = np.array([[0, 1, 1, 2], [1, 0, 2, 1]], dtype=np.longlong)
    edge_weight = np.array([1.0, 2.0, 3.0, 4.0])
    n_nodes = 3
    return edge_index, edge_weight, n_nodes


def test_degree(sample_graph):
    """
    Test cases for the degree function.
    """
    edge_index, edge_weight, n_nodes = sample_graph

    # Test without edge weights
    expected_degree_unweighted = np.array([1.0, 2.0, 1.0])
    calculated_degree_unweighted = degree(edge_index, n_nodes=n_nodes)
    assert np.allclose(
        calculated_degree_unweighted, expected_degree_unweighted
    ), "Unweighted degree calculation is incorrect"

    # Test with edge weights
    expected_degree_weighted = np.array([1.0, 5.0, 4.0])
    calculated_degree_weighted = degree(edge_index, edge_weight, n_nodes)
    assert np.allclose(
        calculated_degree_weighted, expected_degree_weighted
    ), "Weighted degree calculation is incorrect"

    # Test with inferred n_nodes
    calculated_degree_inferred_nodes = degree(edge_index, edge_weight)
    assert np.allclose(
        calculated_degree_inferred_nodes, expected_degree_weighted
    ), "Degree calculation with inferred n_nodes is incorrect"

    # Test with empty graph
    empty_index = np.empty((2, 0), dtype=np.longlong)
    expected_empty_degree = np.zeros(0)
    calculated_empty_degree = degree(empty_index, n_nodes=0)
    assert np.allclose(calculated_empty_degree, expected_empty_degree), "Degree calculation with empty graph is incorrect"



def test_add_self_loops(sample_graph):
    """
    Test cases for the add_self_loops function.
    """
    edge_index, edge_weight, n_nodes = sample_graph
    # Test with edge weights
    new_edge_index_weighted, new_edge_weight_weighted = add_self_loops(
        edge_index, edge_weight, n_nodes
    )
    expected_edge_index_weighted = np.array(
        [[0, 1, 1, 2, 0, 1, 2], [1, 0, 2, 1, 0, 1, 2]], dtype=np.longlong
    )
    expected_edge_weight_weighted = np.array([1.0, 2.0, 3.0, 4.0, 1.0, 1.0, 1.0])
    assert np.array_equal(
        new_edge_index_weighted, expected_edge_index_weighted
    ), "Self-loops with weights: edge index incorrect"
    assert np.allclose(
        new_edge_weight_weighted, expected_edge_weight_weighted
    ), "Self-loops with weights: edge weights incorrect"

    # Test without edge weights
    new_edge_index_unweighted, new_edge_weight_unweighted = add_self_loops(
        edge_index, n_nodes=n_nodes
    )

    expected_edge_index_unweighted = np.array(
        [[0, 1, 1, 2, 0, 1, 2], [1, 0, 2, 1, 0, 1, 2]], dtype=np.longlong
    )
    expected_edge_weight_unweighted = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
    assert np.array_equal(
        new_edge_index_unweighted, expected_edge_index_unweighted
    ), "Self-loops without weights: edge index incorrect"
    assert np.allclose(
        new_edge_weight_unweighted, expected_edge_weight_unweighted
    ), "Self-loops without weights: edge weights incorrect"

    # Test with empty graph
    empty_index = np.empty((2, 0), dtype=np.longlong)
    new_edge_index_empty, new_edge_weight_empty = add_self_loops(empty_index, n_nodes=0)
    expected_empty_index = np.empty((2, 0), dtype=np.longlong)
    expected_empty_weight = np.empty(0)
    assert np.array_equal(
        new_edge_index_empty, expected_empty_index
    ), "Self-loops with empty graph: edge index incorrect"
    assert np.allclose(
        new_edge_weight_empty, expected_empty_weight
    ), "Self-loops with empty graph: edge weights incorrect"



def test_normalize_adj(sample_graph):
    """
    Test cases for the normalize_adj function.
    """
    edge_index, edge_weight, n_nodes = sample_graph
    # Test with edge weights
    normalized_edge_weight_weighted = normalize_adj(edge_index, edge_weight, n_nodes)
    expected_normalized_edge_weight_weighted = np.array(
        [1 / np.sqrt(1 * 5), 2 / np.sqrt(5 * 1), 3 / np.sqrt(5 * 4), 4 / np.sqrt(4 * 5)]
    )
    assert np.allclose(
        normalized_edge_weight_weighted, expected_normalized_edge_weight_weighted
    ), "Weighted normalization is incorrect"

    # Test without edge weights
    normalized_edge_weight_unweighted = normalize_adj(edge_index, n_nodes=n_nodes)
    expected_normalized_edge_weight_unweighted = np.array(
        [1 / np.sqrt(1 * 2), 1 / np.sqrt(2 * 1), 1 / np.sqrt(2 * 1), 1 / np.sqrt(1 * 2)]
    )
    assert np.allclose(
        normalized_edge_weight_unweighted, expected_normalized_edge_weight_unweighted
    ), "Unweighted normalization is incorrect"

    # Test with empty graph
    empty_index = np.empty((2, 0), dtype=np.longlong)
    normalized_edge_weight_empty = normalize_adj(empty_index, n_nodes=0)
    expected_empty_weight = np.array([])
    assert np.allclose(
        normalized_edge_weight_empty, expected_empty_weight
    ), "Normalization with empty graph is incorrect"



def test_build_adj():
    """
    Test cases for the build_adj function.
    """
    # Test case 1: Basic test with coordinates and features
    coords1 = np.array([[0, 0], [1, 0], [0, 1]], dtype=np.float32)
    feat1 = np.array([[1, 2], [3, 4], [5, 6]], dtype=np.float32)
    dist_thr1 = 1.5
    edge_index1, edge_weight1 = build_adj(coords1, feat1, dist_thr1)
    expected_edge_index1 = np.array(
        [[0, 0, 1, 1, 2, 2], [1, 2, 0, 2, 0, 1]], dtype=np.longlong
    )
    expected_edge_weight1 = np.array(
        [
            np.exp(-np.linalg.norm(feat1[0] - feat1[1]) / 2),
            np.exp(-np.linalg.norm(feat1[0] - feat1[2]) / 2),
            np.exp(-np.linalg.norm(feat1[1] - feat1[0]) / 2),
            np.exp(-np.linalg.norm(feat1[1] - feat1[2]) / 2),
            np.exp(-np.linalg.norm(feat1[2] - feat1[0]) / 2),
            np.exp(-np.linalg.norm(feat1[2] - feat1[1]) / 2),
        ]
    )
    assert np.array_equal(edge_index1, expected_edge_index1), "Test case 1: edge index incorrect"
    assert np.allclose(edge_weight1, expected_edge_weight1), "Test case 1: edge weights incorrect"

    # Test case 2: Without features
    coords2 = np.array([[0, 0], [1, 0], [0, 1]], dtype=np.float32)
    dist_thr2 = 1.5
    edge_index2, edge_weight2 = build_adj(coords2, dist_thr=dist_thr2)
    expected_edge_index2 = np.array(
        [[0, 0, 1, 1, 2, 2], [1, 2, 0, 2, 0, 1]], dtype=np.longlong
    )
    expected_edge_weight2 = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
    assert np.array_equal(edge_index2, expected_edge_index2), "Test case 2: edge index incorrect"
    assert np.allclose(edge_weight2, expected_edge_weight2), "Test case 2: edge weights incorrect"

    # Test case 3: With self-loops
    coords3 = np.array([[0, 0], [1, 0], [0, 1]], dtype=np.float32)
    feat3 = np.array([[1, 2], [3, 4], [5, 6]], dtype=np.float32)
    dist_thr3 = 1.5
    add_self_loops3 = True
    edge_index3, edge_weight3 = build_adj(coords3, feat3, dist_thr3, add_self_loops3)
    expected_edge_index3 = np.array(
        [
            [0, 0, 0, 1, 1, 1, 2, 2, 2],
            [0, 1, 2, 1, 0, 2, 2, 0, 1],
        ],
        dtype=np.longlong,
    )
    expected_edge_weight3 = np.array(
        [
            1.0,
            np.exp(-np.linalg.norm(feat3[0] - feat3[1]) / 2),
            np.exp(-np.linalg.norm(feat3[0] - feat3[2]) / 2),
            1.0,
            np.exp(-np.linalg.norm(feat3[1] - feat3[0]) / 2),
            np.exp(-np.linalg.norm(feat3[1] - feat3[2]) / 2),
            1.0,
            np.exp(-np.linalg.norm(feat3[2] - feat3[0]) / 2),
            np.exp(-np.linalg.norm(feat3[2] - feat3[1]) / 2),
        ]
    )
    assert np.array_equal(edge_index3, expected_edge_index3), "Test case 3: edge index incorrect"
    assert np.allclose(edge_weight3, expected_edge_weight3), "Test case 3: edge weights incorrect"

    # Test case 4: No neighbors within distance threshold
    coords4 = np.array([[0, 0], [10, 0], [0, 10]], dtype=np.float32)
    feat4 = np.array([[1, 2], [3, 4], [5, 6]], dtype=np.float32)
    dist_thr4 = 1.5
    edge_index4, edge_weight4 = build_adj(coords4, feat4, dist_thr4)
    assert edge_index4.size == 0, "Test case 4: edge index incorrect"
    assert edge_weight4.size == 0, "Test case 4: edge weights incorrect"
