import numpy as np
import scipy
import torch
import torch.nn.functional as F
from torch import Tensor


def gaussian_kernel_torch(embs_a, embs_b, sigma=1.0):
    """
    Computes the Gaussian kernel matrix between two sets of embeddings using PyTorch.
    :param embs_a: Tensor of shape (batch_size_a, embedding_dim) containing the first set of embeddings.
    :param embs_b: Tensor of shape (batch_size_b, embedding_dim) containing the second set of embeddings.
    :param sigma: Width of the Gaussian kernel.
    :return: Tensor of shape (batch_size_a, batch_size_b) containing the Gaussian kernel matrix.
    """
    if not isinstance(embs_a, torch.Tensor):
        embs_a = torch.tensor(embs_a)

    if not isinstance(embs_b, torch.Tensor):
        embs_b = torch.tensor(embs_b)

    # Compute the pairwise distances between the embeddings
    dist_matrix = torch.cdist(embs_a, embs_b)

    # Compute the Gaussian kernel matrix
    kernel_matrix = torch.exp(-(dist_matrix**2) / (2 * sigma**2))

    return kernel_matrix


def estimate_pdf(scores: list):
    """
    estimate scores probability density function
    :scores: list of distance scores from topic features to topic centroid
    :return: distribution
    """
    return scipy.stats.gaussian_kde(scores)


def cos_sim(a: Tensor, b: Tensor):
    """
    Computes the cosine similarity cos_sim(a[i], b[j]) for all i and j.
    :return: Matrix with res[i][j]  = cos_sim(a[i], b[j])
    """
    if not isinstance(a, torch.Tensor):
        a = torch.tensor(np.array(a))

    if not isinstance(b, torch.Tensor):
        b = torch.tensor(np.array(b))

    if len(a.shape) == 1:
        a = a.unsqueeze(0)

    if len(b.shape) == 1:
        b = b.unsqueeze(0)

    a_norm = torch.nn.functional.normalize(a, p=2, dim=1)
    b_norm = torch.nn.functional.normalize(b, p=2, dim=1)
    return torch.mm(a_norm, b_norm.transpose(0, 1))


def cos_sim_torch(embs_a: Tensor, embs_b: Tensor) -> Tensor:
    """
    Computes the cosine similarity cos_sim(a[i], b[j]) for all i and j.
    Using torch.nn.functional.cosine_similarity
    :return: Matrix with res[i][j]  = cos_sim(a[i], b[j])
    """
    if not isinstance(embs_a, torch.Tensor):
        embs_a = torch.tensor(np.array(embs_a))

    if not isinstance(embs_b, torch.Tensor):
        embs_b = torch.tensor(np.array(embs_b))

    if len(embs_a.shape) == 1:
        embs_a = embs_a.unsqueeze(0)

    if len(embs_b.shape) == 1:
        embs_b = embs_b.unsqueeze(0)
    A = F.cosine_similarity(embs_a.unsqueeze(1), embs_b.unsqueeze(0), dim=2)
    return A


def prune_ref_docs(qa_embs, ref_embs, ref_docs, threshold=0.1):
    """
    Drops unnecessary documents from the reference embeddings and updates the list of reference documents,
    and then recomputes the adjacency matrix.

    Parameters:
    qa_embs (numpy array): The embedding matrix of QA pairs.
    ref_embs (numpy array): The embedding matrix of reference sentences.
    ref_docs (list): The list of reference documents.
    threshold (float): The threshold below which documents are considered unnecessary.

    Returns:
    pruned_ref_embs (numpy array): The pruned embedding matrix of reference sentences.
    pruned_ref_docs (list): The pruned list of reference documents.
    pruned_A (numpy array): The pruned adjacency matrix.
    """

    # Compute the initial adjacency matrix with full reference embeddings
    A = gaussian_kernel_torch(qa_embs, ref_embs, sigma=0.5)
    print(f"Before: {A.shape}")
    # Compute the row-wise sum of the adjacency matrix
    row_sum = torch.sum(A, dim=0)

    # Identify the indexes of the relevant documents
    relevant_idx = torch.where(row_sum > threshold * row_sum.max())[0]

    # Drop unnecessary rows from the reference embeddings
    pruned_ref_embs = ref_embs[relevant_idx]

    # Update the list of reference documents
    pruned_ref_docs = [ref_docs[i] for i in relevant_idx]

    # Recompute the adjacency matrix with pruned reference embeddings
    pruned_A = gaussian_kernel_torch(qa_embs, pruned_ref_embs, sigma=0.5)
    print(f"After: {pruned_A.shape}")
    return pruned_ref_embs, pruned_ref_docs, pruned_A


def compute_cos_sim_kernel(embs, threshold=0.65, kernel_type="cosine", sigma=1.0):
    # match case to kernel type
    if kernel_type == "gaussian":
        A = gaussian_kernel_torch(embs, embs, sigma)
    if kernel_type == "cosine":
        A = cos_sim(embs, embs)
    adj_matrix = torch.zeros_like(A)
    adj_matrix[A > threshold] = 1
    adj_matrix[A <= threshold] = 0
    adj_matrix = adj_matrix.numpy().astype(np.float32)
    return adj_matrix


def compute_kernel(embs):
    # match case to kernel type
    A = cos_sim(embs, embs)
    adj_matrix = A.numpy().astype(np.float32)
    return adj_matrix


def degree_matrix(A):
    """
    compute degree matrix using adjacency distance matrix from pairwise distances
    :A: nxn size matrix embedding minmaxed using mu sigma and pairwise distances
    :return: degree matrix
    """
    n = A.shape[0]
    D = np.zeros((n, n))
    for i in range(n):
        D[i, i] = np.sum(A[i, :])
    return D


def graph_laplacian(A):
    """
    compute graph laplacian using degree and adjacency matrix from pairwise distances
    :A: nxn size matrix embedding minmaxed using mu sigma and pairwise distances
    :return: graph laplacian, and degree matrix
    """
    D = degree_matrix(A)
    L = D - A
    return L, D
