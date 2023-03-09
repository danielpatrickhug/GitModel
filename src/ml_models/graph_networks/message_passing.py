import numpy as np
import scipy.sparse as sp
from tqdm import tqdm


def k_hop_message_passing(A, node_features, k):
    """
    Compute the k-hop adjacency matrix and aggregated features using message passing.

    Parameters:
    A (numpy array): The adjacency matrix of the graph.
    node_features (numpy array): The feature matrix of the nodes.
    k (int): The number of hops for message passing.

    Returns:
    A_k (numpy array): The k-hop adjacency matrix.
    agg_features (numpy array): The aggregated feature matrix for each node in the k-hop neighborhood.
    """

    print("Compute the k-hop adjacency matrix")
    A_k = np.linalg.matrix_power(A, k)

    print("Aggregate the messages from the k-hop neighborhood:")
    agg_features = node_features.copy()

    for i in tqdm(range(k)):
        agg_features += np.matmul(np.linalg.matrix_power(A, i + 1), node_features)

    return A_k, agg_features


def k_hop_message_passing_sparse(A, node_features, k):
    """
    Compute the k-hop adjacency matrix and aggregated features using message passing.

    Parameters:
    A (numpy array or scipy sparse matrix): The adjacency matrix of the graph.
    node_features (numpy array or scipy sparse matrix): The feature matrix of the nodes.
    k (int): The number of hops for message passing.

    Returns:
    A_k (numpy array): The k-hop adjacency matrix.
    agg_features (numpy array): The aggregated feature matrix for each node in the k-hop neighborhood.
    """

    # Convert input matrices to sparse matrices if they are not already
    if not sp.issparse(A):
        A = sp.csr_matrix(A)
    if not sp.issparse(node_features):
        node_features = sp.csr_matrix(node_features)

    # Compute the k-hop adjacency matrix and the aggregated features
    A_k = A.copy()
    agg_features = node_features.copy()

    for i in tqdm(range(k)):
        # Compute the message passing for the k-hop neighborhood
        message = A_k.dot(node_features)
        # Apply a GCN layer to aggregate the messages
        agg_features = A_k.dot(agg_features) + message
        # Update the k-hop adjacency matrix by adding new edges
        A_k += A_k.dot(A)

    return A_k.toarray(), agg_features.toarray()
