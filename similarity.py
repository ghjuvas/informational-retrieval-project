'''
Module for computing similarity between documents and queries.
'''
from typing import Union
import numpy as np
import pandas as pd
from scipy.sparse._csr import csr_matrix
from sklearn.metrics.pairwise import cosine_similarity


def compute_cosine_similarity(query_vector: Union[csr_matrix, np.ndarray],
                              index: Union[csr_matrix, np.ndarray]) -> Union[csr_matrix, np.ndarray]:
    '''
    Compute similarity between query and documents by cosine similarity.
    '''
    similarities = cosine_similarity(index, query_vector)

    return similarities


def compute_dot_product(query_vector: Union[csr_matrix, np.ndarray],
                        index: Union[csr_matrix, np.ndarray]) -> np.ndarray:
    '''
    Compute similarity between query and documents by dot product.
    '''
    similarities = np.dot(index, query_vector.T)

    return similarities


def sort_scores(similarities: Union[csr_matrix, np.ndarray], docs: list) -> pd.DataFrame:
    '''
    Sort scores after computing similarities.
    '''
    if not isinstance(similarities, np.ndarray):
        similarities = similarities.toarray()
    sorted_scores = np.argsort(similarities, axis=0)[::-1]  # arg -> index
    docs_array = np.array(docs)
    sorted_docs = docs_array[sorted_scores.ravel()][:10]
    df = pd.DataFrame(
        sorted_docs,
        index=np.array(range(1, len(sorted_docs)+1)),
        columns=['Ответы на Ваш запрос'])

    return df
