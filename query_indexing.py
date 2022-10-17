'''
Module for query encoding.
'''
from typing import Union
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from scipy.sparse._csr import csr_matrix
import numpy as np
import torch
from transformers import models


def get_query_vector(query: str, vectorizer: Union[CountVectorizer, TfidfVectorizer]) -> csr_matrix:
    '''
    Get vector of the query.
    '''
    query_vector = vectorizer.transform([query])

    return query_vector


def bert_encoding(texts: Union[list, str],
                  tokenizer: models.bert.tokenization_bert_fast.BertTokenizerFast,
                  model: models.bert.modeling_bert.BertModel) -> np.ndarray:
    '''
    Compute inverted index by BERT encoder in the form of a matrix.
    '''

    # tokenize sentences (without preprocessing)
    encoded_input = tokenizer(
        texts, padding=True,
        truncation=True, max_length=24,
        return_tensors='pt')

    # compute token embeddings
    with torch.no_grad():
        model_output = model(**encoded_input)

    # pooling
    token_embeddings = model_output[0] # first element of model_output contains all token embeddings
    input_mask_expanded = encoded_input['attention_mask'].unsqueeze(-1).expand(token_embeddings.size()).float()
    sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
    sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    sentence_embeddings = sum_embeddings / sum_mask
    sentence_embeddings = sentence_embeddings.detach().cpu().numpy()  # to array

    return sentence_embeddings
