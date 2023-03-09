import json
import math

import numpy as np
import openai
import scipy
import scipy.sparse as sp
import torch
from sentence_transformers import SentenceTransformer
from torch import Tensor
from tqdm import tqdm

from ast_parsers.python_ast_parser import get_methods, parse_github_repo
from system_prompts.format_system_prompts import format_system_prompts, format_system_prompts_with_tree


def embed_data(data, key="query", model_name="all-MiniLM-L6-v2", cores=1, gpu=False, batch_size=128):
    """
    Embed the sentences/text using the MiniLM language model (which uses mean pooling)
    """
    print("Embedding data")
    model = SentenceTransformer(model_name)
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
        embeddings = model.encode_multi_process(unique_sentences, pool, batch_size=batch_size, chunk_size=chunk_size)
        model.stop_multi_process_pool(pool)

    print("Embeddings computed")

    mapping = {sentence: embedding for sentence, embedding in zip(unique_sentences, embeddings)}
    embeddings = np.array([mapping[sentence] for sentence in sentences])

    return embeddings


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


def estimate_pdf(scores: list):
    """
    estimate scores probability density function
    :scores: list of distance scores from topic features to topic centroid
    :return: distribution
    """
    return scipy.stats.gaussian_kde(scores)


def chat_gpt_inference(messages: list):
    response = openai.ChatCompletion.create(model="gpt-3.5-turbo", messages=messages, max_tokens=400, temperature=0.4)
    return response


def create_prompt_message_template(text, role="user"):
    if role not in ["user", "assistant"]:
        raise ValueError("Not a valid role. Please use 'user' or 'assistant'.")
    return {"role": role, "content": text}


def compose_inference(text_block, messages):
    user_template = create_prompt_message_template(text_block, role="user")
    messages.append(user_template)
    chat_resp = chat_gpt_inference(messages)
    reply_text = chat_resp["choices"][0]["message"]["content"]
    assistant_template = create_prompt_message_template(reply_text, role="assistant")
    messages.append(assistant_template)
    return messages, reply_text


def process_transcript(segments, file_name, git_repo_path, output_file_path, system_prompt, task, code_type):
    messages = [{"role": "system", "content": system_prompt}]

    with open(output_file_path, "a") as f:
        for i, sent in enumerate(segments):
            text_block = f"""```{sent}```"""
            try:
                messages, reply_text = compose_inference(text_block[:2000], messages)

            except Exception:
                messages, reply_text = compose_inference(
                    text_block[:2000], [{"role": "system", "content": system_prompt}]
                )
            row = {
                "git_repo_path": git_repo_path,
                "file_name": file_name,
                "code_type": code_type,
                "system_task": task,
                "system_prompt": system_prompt,
                "conversation_history": messages,
                "assistant_reply": reply_text,
            }
            json.dump(row, f)
            f.write("\n")

    return messages


def get_repo_contents(git_repo_path):
    contents = parse_github_repo(git_repo_path)
    print(len(contents))
    pruned_contents = []
    for cont in contents:
        fp = cont["file_name"]
        fn = fp.split("/")[-1]
        fn_ = fn.split(".")[0]
        if fn_ in ["__init__"] or fn_.split("_")[-1] in ["test"]:
            continue
        else:
            print(cont["file_name"])
            pruned_contents.append(cont)
    return pruned_contents


def decompose_repo(contents, git_repo_path, name_id, topic_tree, out_path, with_tree=False):
    # This is bad
    for cont in contents:
        if with_tree:
            system_prompts = format_system_prompts(git_repo_path, cont["file_name"])
        else:
            system_prompts = format_system_prompts_with_tree(git_repo_path, cont["file_name"], topic_tree)
        for k, v in zip(system_prompts.keys(), system_prompts.values()):
            func_task = k
            out_file_name = f"{name_id}_{func_task}"
            print(f"file_name: {cont['file_name']}")
            num_funcs = len(cont["functions"])
            num_classes = len(cont["classes"])
            print(f"Imports: {cont['imports']}")
            try:
                if num_funcs > 0 or num_classes > 0:
                    print(f"functions: {cont['functions']}")
                    _ = process_transcript(
                        cont["functions"],
                        cont["file_name"],
                        git_repo_path,
                        f"{out_path}/{out_file_name}.jsonl",
                        system_prompts[func_task],
                        func_task,
                        "functions",
                    )
                    print(f"Classes: {cont['classes']}")
                    _ = process_transcript(
                        cont["classes"],
                        cont["file_name"],
                        git_repo_path,
                        f"{out_path}/{out_file_name}.jsonl",
                        system_prompts[func_task],
                        func_task,
                        "classes",
                    )
                    for cls in cont["classes"]:
                        cls_funcs = get_methods(cls)

                        print(f"len of class: {len(cls)}")
                        for method in cls_funcs:
                            print(f"len of method: {len(method)}")
                        _ = process_transcript(
                            cls_funcs,
                            cont["file_name"],
                            git_repo_path,
                            f"{out_path}/{out_file_name}.jsonl",
                            system_prompts[func_task],
                            func_task,
                            "methods",
                        )
            except Exception as e:
                print(e)
                continue
