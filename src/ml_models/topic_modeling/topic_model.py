import os
from typing import Tuple

import numpy as np
import pandas as pd
from bertopic import BERTopic
from bertopic.representation import MaximalMarginalRelevance, OpenAI
from bertopic.vectorizers import ClassTfidfTransformer
from sentence_transformers import SentenceTransformer
from sklearn.cluster import AgglomerativeClustering
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import CountVectorizer

os.environ["TOKENIZERS_PARALLELISM"] = "false"


class TopicModel:
    def __init__(self, config):
        self.config = config

    def __repr__(self) -> str:
        return f"TopicModel(config={self.config})"

    def fit_topic_model(self, topic_model, data, embeddings, key="query"):
        topics, probs = topic_model.fit_transform(data[key].to_list(), embeddings)
        return topics, probs

    def get_topic_info(self, topic_model):
        return topic_model.get_topic_info()

    def reduce_topics(self, topic_model, data, nr_topics, key="query"):
        topic_model.reduce_topics(data[key].to_list(), nr_topics)
        return topic_model

    def get_representative_docs(self, topic_model):
        return topic_model.get_representative_docs()

    def reduce_outliers(self, topic_model, data, topics, probs, key="query", strategy="c-tf-idf"):
        if strategy == "c-tf-idf":
            new_topics = topic_model.reduce_outliers(data[key].to_list(), topics, strategy, threshold=0.1)
        elif strategy == "embeddings":
            new_topics = topic_model.reduce_outliers(data[key].to_list(), topics, strategy)
        elif strategy == "distributions":
            new_topics = topic_model.reduce_outliers(
                data[key].to_list(), topics, probabilities=probs, strategy=strategy
            )
        else:
            raise ValueError("Invalid strategy")
        topic_model.update_topics(
            data[key].to_list(),
            topics=new_topics,
            representation_model=self.representation_model,
            vectorizer_model=self.vectorizer_model,
            ctfidf_model=self.ctfidf_model,
        )
        return topic_model, new_topics

    def compute_hierarchical_topic_tree(self, topic_model, data, key="query"):
        hierarchical_topics = topic_model.hierarchical_topics(data[key].to_list())
        tree = topic_model.get_topic_tree(hierarchical_topics)
        return hierarchical_topics, tree

    def run(self, data, gnn_outputs: Tuple[np.ndarray, np.ndarray], key="query") -> dict:
        """
        Run the topic modeling.
        """
        # Prepare the data
        if self.config["auto_cluster"]:
            topic_model = BERTopic(
                nr_topics="auto",
                vectorizer_model=eval(self.config["vectorizer_model"]),
                embedding_model=self.config["embedding_model"],
                representation_model=eval(self.config["representation_model"]),
                n_gram_range=(1, 2),
                min_topic_size=10,
                top_n_words=10,
                calculate_probabilities=False,
            )
        else:
            # Fit the BERTopic model
            topic_model = BERTopic(
                nr_topics="auto",
                vectorizer_model=eval(self.config["vectorizer_model"]),
                umap_model=eval(self.config["dimensionality_reduction"]),
                hdbscan_model=eval(self.config["clustering_model"]),
                embedding_model=eval(self.config["embedding_model"]),
                representation_model=eval(self.config["representation_model"]),
                n_gram_range=(1, 2),
                min_topic_size=10,
                top_n_words=10,
                calculate_probabilities=False,
            )
        topics, probabilities = topic_model.fit_transform(data[key].to_list(), gnn_outputs[1])
        freq = self.get_topic_info(topic_model)
        rep_docs = self.get_representative_docs(topic_model)
        hr, tree = self.compute_hierarchical_topic_tree(topic_model=topic_model, data=data)
        # add dict with topic info
        docs = []
        for k, v in rep_docs.items():
            docs.append((k, v))
        rep = pd.DataFrame(docs, columns=["Topic", "Rep_docs"])
        topic_info_freq = pd.merge(freq, rep, on="Topic")
        data["topic"] = topics
        data["prob"] = probabilities
        return {"data": data, "topic_info": topic_info_freq, "tree": tree}
