'''
Module for search by chosen index method.
'''
import pandas as pd
import nlp
import query_indexing
import data
import similarity

DOCS = data.read_text_file('answers.txt').split('\n')


def tf_idf_search(query: str, vectorizer, index, docs=DOCS) -> pd.DataFrame:
    '''
    Search by TF-IDF index.
    '''
    query = nlp.processing(query)
    query_vector = query_indexing.get_query_vector(query, vectorizer)
    similarities = similarity.compute_cosine_similarity(query_vector, index)
    df = similarity.sort_scores(similarities, docs)

    return df


def bm25_search(query: str, vectorizer, index, docs=DOCS) -> pd.DataFrame:
    '''
    Search by Okapi BM25 index.
    '''
    query = nlp.processing(query)
    query_vector = query_indexing.get_query_vector(query, vectorizer)
    similarities = similarity.compute_dot_product(query_vector, index)
    df = similarity.sort_scores(similarities, docs)

    return df


def bert_search(query: str, tokenizer, model, index, docs=DOCS) -> pd.DataFrame:
    '''
    Search by BERT index.
    '''
    query_vector = query_indexing.bert_encoding(query, tokenizer, model)
    similarities = similarity.compute_cosine_similarity(query_vector, index)
    df = similarity.sort_scores(similarities, docs)

    return df
