import math
from typing import Tuple

import numpy as np
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
