'''
User interface of the search app.
'''
import time
import streamlit as st
import manage_app
from data import MODELS

st.set_page_config(layout="wide")

# info
title_cols = st.columns([1, 1.8, 1])
title_cols[1].title('💓Поисковик про любовь💓')
info_cols = st.columns([1, 2.2, 1])
info_cols[1].write('Здесь Вы можете найти ответ на любой вопрос, связанный с делами романтическими!')

# choice
st.header('Поиск')

method_col, query_col = st.columns([1, 3])
method = method_col.selectbox(label='Метод', options=['TF-IDF', 'Okapi BM25', 'BERT'])
query = query_col.text_input(label='Ваш запрос', key='query')

search_cols = st.columns([1, 0.1, 1])
search_button = search_cols[1].button(label='Поиск')

# search
if search_button:
    placeholder = st.empty()
    start_time = time.time()
    df = manage_app.main(method, query, MODELS)
    work_time = round((time.time() - start_time), 3)
    if isinstance(df, str) and df == '':
        placeholder.error('Вы ввели пустой запрос!', icon='🚨')
    else:
        placeholder.dataframe(df, use_container_width=True)
        time_placeholder = st.empty()
        time_placeholder.info(f'Поиск был произведён за {work_time} секунд', icon='⌛')

else:
    st.info(
        'Чтобы начать поиск, выберите индекс (BM25 или BERT), введите свой запрос и нажмите на кнопку поиска',
        icon='ℹ️'
    )
