'''
Module for managing actions in the streamlit app.
'''
import pandas as pd
import search
import measurement


@measurement.performance
@measurement.profile
def main(method: str, query: str, models: dict) -> pd.DataFrame:
    '''
    Main function to run search of the documents by chosen method.
    '''
    if query == '':
        return ''

    if method == 'TF-IDF':
        df = search.tf_idf_search(query, models['tfidf_vectorizer'], models['tfidf_index'])

    if method == 'Okapi BM25':
        df = search.bm25_search(query, models['bm25_vectorizer'], models['bm25_index'])

    if method == 'BERT':
        df = search.bert_search(
            query, models['bert_tokenizer'],
            models['bert_model'], models['bert_index']
            )

    return df
