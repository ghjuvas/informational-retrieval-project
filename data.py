'''
Module for open files with the indexes and vectorizers.
'''
import pickle
from typing import Any, Tuple
import numpy as np
from scipy import sparse
from transformers import AutoTokenizer, AutoModel, models
from scipy.sparse._csr import csr_matrix


def load_transformers_models() -> Tuple[models.bert.tokenization_bert_fast.BertTokenizerFast,
                                        models.bert.modeling_bert.BertModel]:
    '''
    Load huggingface models for BERT encoding.
    '''
    tokenizer = AutoTokenizer.from_pretrained("sberbank-ai/sbert_large_nlu_ru")
    model = AutoModel.from_pretrained("sberbank-ai/sbert_large_nlu_ru")

    return tokenizer, model


def read_binary_file(path: str) -> Any:
    '''
    Read binary files with pickle.
    '''
    with open(path, 'rb') as bf:
        pickle_object = pickle.load(bf)

    return pickle_object


def read_text_file(path: str) -> str:
    '''
    Read text file with UTF-8 encoding.
    '''
    with open(path, 'r', encoding='utf-8') as tf:
        text = tf.read()

    return text


def read_sparse_npz(path: str) -> csr_matrix:
    '''
    Get sparse matrix from .npz file.
    '''
    matrix = sparse.load_npz(path)

    return matrix


def read_numpy_npy(path: str) -> np.ndarray:
    '''
    Get numpy array from .npy file.
    '''
    array = np.load(path)

    return array


def load_data() -> dict:
    '''
    Load all the needed data.
    '''
    tfidf_index = read_sparse_npz('tfidf_index.npz')
    tfidf_vectorizer = read_binary_file('tfidf_vectorizer.pkl')
    bm25_index = read_sparse_npz('bm25_index.npz')
    bm25_vectorizer = read_binary_file('bm25_count_vectorizer.pkl')
    bert_tokenizer, bert_model = load_transformers_models()
    bert_index = read_numpy_npy('bert_index.npy')

    models = {
        'tfidf_index': tfidf_index,
        'tfidf_vectorizer': tfidf_vectorizer,
        'bm25_index': bm25_index,
        'bm25_vectorizer': bm25_vectorizer,
        'bert_tokenizer': bert_tokenizer,
        'bert_model': bert_model,
        'bert_index': bert_index
    }

    return models


MODELS = load_data()
