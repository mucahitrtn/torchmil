import numpy as np
from scipy.spatial import KDTree

def degree(
        index: np.ndarray,
        edge_weight: np.ndarray = None,
        n_nodes: int = None
) -> np.ndarray:
    """
    Compute the degree of the adjacency matrix. Assumes that the graph is undirected.

    Arguments:
        index: Edge index of the adjacency matrix, shape (2, n_edges).
        edge_weight: Edge weight of the adjacency matrix, shape (n_edges,).
        n_nodes: Number of nodes in the graph.

    Returns:
        degree: Degree of the adjacency matrix.
    """

    if edge_weight is None:
        edge_weight = np.ones(index.shape[1])

    if n_nodes is None:
        n_nodes = index.max() + 1

    out = np.zeros((n_nodes))
    np.add.at(out, index[0,:], edge_weight)
    return out

def add_self_loops(
        edge_index: np.ndarray,
        edge_weight: np.ndarray = None,
        n_nodes: int = None,
) -> tuple[np.ndarray, np.ndarray]:
        """
        Add self-loops to the adjacency matrix.

        Arguments:
            edge_index: Edge index of the adjacency matrix, shape (2, n_edges).
            edge_weight: Edge weight of the adjacency matrix, shape (n_edges,).
            n_nodes: Number of nodes in the graph.

        Returns:
            new_edge_index: Edge index of the adjacency matrix with self-loops.
            new_edge_weight: Edge weight of the adjacency matrix with self-loops.
        """

        if n_nodes is None:
            n_nodes = edge_index.max() + 1

        if edge_weight is None:
            edge_weight = np.ones(edge_index.shape[1])

        loop_index = np.arange(0, n_nodes)
        loop_index = np.tile(loop_index, (2,1))

        if edge_index.shape[0] == 0:
            new_edge_index = loop_index
            new_edge_weight = np.ones(n_nodes)
        else:
            if edge_weight is None:
                edge_weight = np.ones(edge_index.shape[1])
            new_edge_index = np.hstack([edge_index, loop_index])
            new_edge_weight = np.concatenate([edge_weight, np.ones(n_nodes)])
        return new_edge_index, new_edge_weight


def normalize_adj(
        edge_index: np.ndarray,
        edge_weight: np.ndarray = None,
        n_nodes: int = None,
) -> np.ndarray:
    """
    Normalize the adjacency matrix.

    Arguments:
        edge_index: Edge index of the adjacency matrix.
        edge_weight: Edge weight of the adjacency matrix.
        n_nodes: Number of nodes in the graph.

    Returns:
        edge_weight: Edge weight of the normalized adjacency matrix.
    """

    if edge_weight is None:
        edge_weight = np.ones(edge_index.shape[1])

    if n_nodes is None:
        n_nodes = edge_index.max() + 1

    if edge_index.shape[0] == 0:
        new_edge_weight = np.array([])
    else:
        row = edge_index[0]
        col = edge_index[1]
        deg = degree(edge_index, edge_weight, n_nodes).astype(np.float32)
        with np.errstate(divide='ignore'):
            deg_inv_sqrt = np.power(deg, -0.5)
            deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
        new_edge_weight = deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]

    return new_edge_weight

def build_adj(
    coords: np.ndarray,
    feat: np.ndarray = None,
    dist_thr: float = 1.0,
    add_self_loops: bool = False
) -> tuple[np.ndarray, np.ndarray]:
    """
    Build the adjacency matrix for a general graph given the coordinates and features of the nodes.

    Arguments:
        coords: Coordinates of the nodes.
        feat: Features of the nodes, used to compute the edge weights. If None, the adjacency matrix is binary.
        dist_thr: Distance threshold to consider two nodes as neighbors. Default is 1.0.
        add_self_loops: Whether to add self-loops.

    Returns:
        edge_index: Edge index of the adjacency matrix.
        edge_weight: Edge weight of the adjacency matrix
    """

    kdtree = KDTree(coords)

    # Build adjacency matrix
    n_nodes = len(coords)
    edge_index = []
    edge_weight = []
    for i in range(n_nodes):

        # Self-loop
        if add_self_loops:
            edge_index.append([i, i])
            edge_weight.append(1.0)

        # Find neighboring nodes within dist_thr distance
        neighbors = kdtree.query_ball_point(coords[i], dist_thr)
        for j in neighbors:
            if i != j:
                edge_index.append([i, j])
                if feat is not None:
                    dist = np.exp(-np.linalg.norm(feat[i] -
                                  feat[j]) / feat.shape[1])
                else:
                    dist = 1.0
                edge_weight.append(dist)

    if len(edge_index) == 0:
        edge_index = np.array([[], []]).astype(np.longlong)
        edge_weight = np.array([])
    else:
        edge_index = np.array(edge_index).T.astype(np.longlong)  # (2, n_edges)
        edge_weight = np.array(edge_weight)  # (n_edges,)

    return edge_index, edge_weight
