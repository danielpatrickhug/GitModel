import math

import numpy as np
from sentence_transformers import SentenceTransformer


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
