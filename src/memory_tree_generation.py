from bertopic import BERTopic
from bertopic.representation import OpenAI
from bertopic.vectorizers import ClassTfidfTransformer
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import CountVectorizer


def load_topic_model(diversity=0.1, min_topic_size=10, nr_topics="auto", model_name="all-MiniLM-L6-v2"):
    vectorizer_model = CountVectorizer(stop_words="english")
    ctfidf_model = ClassTfidfTransformer(reduce_frequent_words=True)
    representation_model = OpenAI(model="gpt-3.5-turbo", delay_in_seconds=1, chat=True)
    model = SentenceTransformer(model_name)
    topic_model = BERTopic(
        nr_topics=nr_topics,
        min_topic_size=min_topic_size,
        representation_model=representation_model,
        vectorizer_model=vectorizer_model,
        ctfidf_model=ctfidf_model,
        embedding_model=model,
    )
    return topic_model


def fit_topic_model(topic_model, data, embeddings, key="query"):
    topics, probs = topic_model.fit_transform(data[key].to_list(), embeddings)
    return topics, probs


def get_topic_info(topic_model):
    return topic_model.get_topic_info()


def reduce_topics(topic_model, data, nr_topics, key="query"):
    topic_model.reduce_topics(data[key].to_list(), nr_topics)
    return topic_model


def get_representative_docs(topic_model):
    return topic_model.get_representative_docs()


def reduce_outliers(topic_model, data, topics, probs, key="query", strategy="c-tf-idf"):
    vectorizer_model = CountVectorizer(stop_words="english")
    ctfidf_model = ClassTfidfTransformer(reduce_frequent_words=True)
    representation_model = OpenAI(model="gpt-3.5-turbo", delay_in_seconds=1, chat=True)
    # representation_model = MaximalMarginalRelevance(diversity=diversity)
    if strategy == "c-tf-idf":
        new_topics = topic_model.reduce_outliers(data[key].to_list(), topics, strategy, threshold=0.1)
    elif strategy == "embeddings":
        new_topics = topic_model.reduce_outliers(data[key].to_list(), topics, strategy)
    elif strategy == "distributions":
        new_topics = topic_model.reduce_outliers(data[key].to_list(), topics, probabilities=probs, strategy=strategy)
    else:
        raise ValueError("Invalid strategy")
    topic_model.update_topics(
        data[key].to_list(),
        topics=new_topics,
        representation_model=representation_model,
        vectorizer_model=vectorizer_model,
        ctfidf_model=ctfidf_model,
    )
    return topic_model, new_topics


def compute_hierarchical_topic_tree(topic_model, data, key="query"):
    hierarchical_topics = topic_model.hierarchical_topics(data[key].to_list())
    tree = topic_model.get_topic_tree(hierarchical_topics)
    return hierarchical_topics, tree
