import pandas as pd
from collections import Counter

import nltk
from nltk.tokenize import sent_tokenize

nltk.download("punkt_tab", quiet=True)


def split_sentences(articles: list[str]) -> tuple[list[int], list[str]]:
    """
    Split news articles in sentences tracking original article for each sentence.
    """

    sentences = [
        sent_tokenize(str(article), language="italian") for article in articles
    ]
    # track the document ID for all sentences
    doc_ids = [[idx] * len(s) for idx, s in enumerate(sentences)]
    sentences = [sentence for doc in sentences for sentence in doc]
    doc_ids = [idx for l in doc_ids for idx in l]

    return doc_ids, sentences


def group_topics_by_doc(
    doc_ids: list[int], sentences: list[str], topics: list[str], topics_names: dict
) -> pd.Series:
    """
    Group topics by news article.
    Parameters:
        doc_ids: a list of documents (news) ids, to track news article for each sentence
        sentences: a list of sentences extracted from news articles
        topics: a list containing sentences topics
        topics_names: a mapping from id to topic name

    Returns:
        pd.Series: a list of topics for each news articles
    """

    sent_topics = pd.DataFrame(
        {"doc_id": doc_ids, "sentences": sentences, "topic": topics}
    )

    # Group by article and get the most frequent topic per article
    # Remove outliers (-1) and get most common topics per document
    news_topics = (
        sent_topics[sent_topics["topic"] != -1]
        .groupby("doc_id")["topic"]
        .apply(lambda x: [topics_names[topic] for topic, _ in Counter(x).most_common()])
    )

    return news_topics
    # Merge with the original dataset
    # df["topics"] = df.index.map(article_topics)


def group_sent_by_topics(
    sentences: list[str], topics: list[str], topics_names: dict[int, str]
) -> pd.DataFrame:
    """
    group sentences by topics
    """
    df = pd.DataFrame(
        {"sentences": sentences, "topics": [topics_names[topic] for topic in topics]}
    )

    grouped_df = df.groupby("topics")["sentences"].apply(list).reset_index()

    return grouped_df


def n_topic_per_doc(topics: list[list[str]]) -> pd.DataFrame:
    """
    Calculate in how many articles each topic occurs
    """
    # Flatten the list of topics across all articles
    all_topics = [topic for topic_list in topics.dropna() for topic in topic_list]

    # Count occurrences of each topic
    topic_counts = Counter(all_topics)

    # Convert to a DataFrame for better readability
    topic_counts_df = pd.DataFrame(
        topic_counts.items(), columns=["Topic", "Article_Count"]
    ).sort_values(by="Article_Count", ascending=False)
    return topic_counts_df
