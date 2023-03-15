import math
import multiprocessing
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer

from src.config import Config
from src.ml_models.graph_networks.kernels import compute_kernel_by_type, graph_laplacian
from src.ml_models.graph_networks.message_passing import k_hop_message_passing, k_hop_message_passing_sparse


class GNNHead:
    def __init__(self, config: Config):
        self.config = config

    def __repr__(self) -> str:
        return f"GNNHead(config={self.config})"

    def compute_kernel_matrix(self, node_features: np.ndarray) -> np.ndarray:
        """
        Compute the kernel matrix using the specified kernel function.

        Parameters:
        node_features (numpy array): The feature matrix of the nodes.

        Returns:
        kernel_matrix (numpy array): The kernel matrix computed using the specified kernel function.
        """
        if self.config["kernel_id"] == "cosine_similarity":
            kernel_matrix = compute_kernel_by_type(
                node_features, threshold=self.config["connection_threshold"], kernel_type="cosine"
            )
        elif self.config["kernel_id"] == "gaussian":
            kernel_matrix = compute_kernel_by_type(
                node_features,
                threshold=self.config["connection_threshold"],
                kernel_type="gaussian",
                sigma=self.config["sigma"],
            )
        else:
            raise ValueError("Invalid kernel function specified.")
        return kernel_matrix

    def embed_data(self, data, key="query", cores=1, gpu=False, batch_size=128):
        """
        Embed the sentences/text using the MiniLM language model (which uses mean pooling)
        """
        print("Embedding data")
        model = SentenceTransformer(self.config["embbedding_model"])
        print("Model loaded")

        sentences = data[key].tolist()
        unique_sentences = data[key].unique()
        print("Unique sentences", len(unique_sentences))

        if cores == 1:
            embeddings = model.encode(unique_sentences, show_progress_bar=True, batch_size=batch_size)
        else:
            devices = ["cpu"] * cores
            if gpu:
                devices = None  # use all CUDA devices

            # Start the multi-process pool on multiple devices
            print("Multi-process pool starting")
            pool = model.start_multi_process_pool(devices)
            print("Multi-process pool started")

            chunk_size = math.ceil(len(unique_sentences) / cores)

            # Compute the embeddings using the multi-process pool
            embeddings = model.encode_multi_process(
                unique_sentences, pool, batch_size=batch_size, chunk_size=chunk_size
            )
            model.stop_multi_process_pool(pool)

        print("Embeddings computed")

        mapping = {sentence: embedding for sentence, embedding in zip(unique_sentences, embeddings)}
        embeddings = np.array([mapping[sentence] for sentence in sentences])

        return embeddings

    def generate_graph(self, data) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate a graph using the kernel function and message passing.

        Parameters:
        node_features (numpy array): The feature matrix of the nodes.

        Returns:
        A_k (numpy array): The k-hop adjacency matrix.
        agg_features (numpy array): The aggregated feature matrix for each node in the k-hop neighborhood.
        """
        embs = self.embed_data(data)
        kernel_matrix = self.compute_kernel_matrix(embs)
        if self.config["graph_id"] == "adjacency":
            A = kernel_matrix
        elif self.config["graph_id"] == "laplacian":
            L, D = graph_laplacian(kernel_matrix)
            A = np.linalg.pinv(D) @ L
        else:
            raise ValueError("Invalid graph type specified.")

        if self.config["is_sparse"]:
            A_k, agg_features = k_hop_message_passing_sparse(A, embs, self.config["k_hop"])
        else:
            A_k, agg_features = k_hop_message_passing(A, embs, self.config["k_hop"])

        return A_k, agg_features


class GATLayer(nn.Module):
    """
    A single layer of the Graph Attention Network.
    """

    def __init__(self, in_features, out_features, dropout=0.0):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.W = nn.Linear(in_features, out_features, bias=False)
        self.a = nn.Linear(2 * out_features, 1, bias=False)

        self.dropout = nn.Dropout(dropout)

    def forward(self, h, adj):
        Wh = self.W(h)
        a_input = self.prepare_attention_input(Wh)
        e = F.leaky_relu(self.a(a_input))
        attention = F.softmax(e, dim=1)
        attention = self.dropout(attention)

        h_prime = torch.matmul(attention.transpose(1, 2), Wh)
        return h_prime

    def prepare_attention_input(self, Wh):
        N = Wh.size()[0]
        Wh_repeated_in_chunks = Wh.repeat_interleave(N, dim=0)
        Wh_repeated_alternating = Wh.repeat(N, 1)
        all_combinations_matrix = torch.cat([Wh_repeated_in_chunks, Wh_repeated_alternating], dim=1)
        return all_combinations_matrix.view(N, N, 2 * self.out_features)


class GAT(nn.Module):
    """
    The Graph Attention Network model.
    """

    def __init__(self, in_features, out_features, dropout=0.0):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.layers = nn.ModuleList(
            [
                GATLayer(in_features, out_features, dropout=dropout),
                GATLayer(out_features, out_features, dropout=dropout),
            ]
        )

    def forward(self, embs, adj_matrices):
        h = embs
        for layer, adj in zip(self.layers, adj_matrices):
            h = layer(h, adj)
        return h


def combine_graphs_with_gat(graphs):
    """
    Combine a list of graphs using the Graph Attention Network.

    Parameters:
    graphs (List of Tuple): List of tuples where each tuple contains an adjacency matrix and its
        corresponding node embeddings.

    Returns:
    combined_adj_matrix (numpy array): The combined adjacency matrix.
    combined_embs (numpy array): The combined node embeddings.
    """
    in_features = graphs[0][1].shape[1]  # input dimensionality of the embeddings
    out_features = 256  # output dimensionality of the embeddings
    gat = GAT(in_features, out_features)

    adj_matrices = [t[0] for t in graphs]
    embs = torch.cat([torch.FloatTensor(t[1]) for t in graphs], dim=0)

    with torch.no_grad():
        combined_embs = gat(embs, adj_matrices)

    combined_adj_matrix = sum(adj_matrices)
    max_value = np.max(combined_adj_matrix)
    normalized_matrix = combined_adj_matrix / max_value
    multiprocessing.resource_tracker.unregister()

    return normalized_matrix, combined_embs.numpy()


def linearly_sum_gnn_heads(
    matrices: List[Tuple[np.ndarray, np.ndarray]], normalization_fn: str = "max"
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Linearly sum a list of adjacency matrices and normalize the result. Also computes the average of the node embeddings
    and checks that the number of documents in the adjacency matrices matches the number of documents in the embeddings set.

    Parameters:
    matrices (List of Tuple): List of tuples where each tuple contains an adjacency matrix and its
        corresponding node embeddings.

    Returns:
    normalized_matrix (numpy array): The linearly summed and normalized adjacency matrix.
    avg_embeddings (numpy array): The average node embeddings.
    """
    if len(matrices) == 1:
        return matrices[0]
    # Compute the average of the embeddings
    avg_embeddings = np.mean([t[1] for t in matrices], axis=0)

    # Check that the number of documents in the adjacency matrices matches the number of documents in the embeddings set
    num_docs_adj = matrices[0][0].shape[0]
    num_docs_embs = avg_embeddings.shape[0]
    if num_docs_adj != num_docs_embs:
        raise ValueError(
            "Number of documents in adjacency matrix does not match number of documents in embeddings set."
        )

    # Linearly sum the adjacency matrices and normalize the matrix by dividing it by the maximum value
    summed_matrix = sum([t[0] for t in matrices])
    if normalization_fn == "max":
        max_value = np.max(summed_matrix)
        normalized_matrix = summed_matrix / max_value
    elif normalization_fn == "sum":
        normalized_matrix = summed_matrix / len(matrices)
    elif normalization_fn == "min-max":
        min_value = np.min(summed_matrix)
        max_value = np.max(summed_matrix)
        normalized_matrix = (summed_matrix - min_value) / (max_value - min_value)
    elif normalization_fn == "z_score":
        mean = np.mean(summed_matrix)
        std = np.std(summed_matrix)
        normalized_matrix = (summed_matrix - mean) / std
    elif normalization_fn == "circular":
        min_value = np.min(summed_matrix)
        max_value = np.max(summed_matrix)
        normalized_matrix_linear = (summed_matrix - min_value) / (max_value - min_value)
        normalized_matrix = np.sin(2 * np.pi * normalized_matrix_linear)
    else:
        raise ValueError(f"Normalization function {normalization_fn} not supported.")

    A_k, agg_embs = k_hop_message_passing_sparse(normalized_matrix, avg_embeddings, 2)
    print(f"adj matrix shape: {A_k.shape}")
    print(f"agg embeddings shape: {agg_embs.shape}")
    return A_k, agg_embs
