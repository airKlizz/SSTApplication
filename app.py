
import streamlit as st
from st_utils import rerun

import tensorflow as tf
import numpy as np
from transformers import TFAutoModel, TFAutoModelWithLMHead, AutoTokenizer
from MsMarco.use import Ranker

from tinydb import TinyDB, Query

_MODEL_NAME = 'roberta-large'
_WEIGTH_PATH = 'MsMarco/model/saved_weights/model_roberta-large_mrr_0.303.h5'

_BASKET_FILENAME = 'basket.db'

@st.cache(hash_funcs={Ranker: hash})
def load_ranker():
  ranker = Ranker('', _MODEL_NAME, _WEIGTH_PATH)
  return ranker

@st.cache(allow_output_mutation=True)
def load_summarizer():
  model = TFAutoModelWithLMHead.from_pretrained("t5-base")
  tokenizer = AutoTokenizer.from_pretrained("t5-base")
  return {'model': model, 'tokenizer': tokenizer}

@st.cache(hash_funcs={Ranker: hash})
def get_passages(ranker, title, num_urls, top_n, top_n_bm25):
  ranker.topic = title
  ranker.find_passages(num_urls)
  top, scores = ranker.get_rerank_top(top_n=top_n, top_n_bm25=top_n_bm25)
  return [{'text': passage.text, 'source': passage.source} for passage in top]

def summarize(summarizer, document, max_length, min_length):
  tokenizer = summarizer['tokenizer']
  model = summarizer['model']
  inputs = tokenizer.encode("summarize: " + document, return_tensors="tf", max_length=512)
  outputs = model.generate(inputs, max_length=max_length, min_length=min_length, length_penalty=2.0, num_beams=4, early_stopping=True)
  return tokenizer.decode(outputs[0], skip_special_tokens=True, clean_up_tokenization_spaces=False)

ranker = load_ranker()
summarizer = load_summarizer()

st.sidebar.title('Semantic Storytelling Application')
st.sidebar.header('Create an news article automaticly')

option = st.sidebar.selectbox('Menu:',
                      ('Passages selection', 'Passages selected', 'Summarization')
                    )

basket = TinyDB(_BASKET_FILENAME).table('basket')

if option == 'Passages selection':
  st.header("Select passages for your article")
  num_urls = st.slider('Number of url to look at:', 1, 50, 5)
  top_n_bm25 = st.slider('Number of passages to re-rank:', 10, 200, 50)
  top_n = st.slider('Number of passages to display:', 1, 50, 10)
  number_in_basket = st.empty()
  st.markdown('*****')
  title = st.text_input('Research article:', '')
  if title != '':
    passages = get_passages(ranker, title, num_urls, top_n, top_n_bm25)
    for passage in passages:
      passage_hash = hash(passage['text']+passage['source'])
      if st.checkbox('Select', key='{}'.format(passage_hash)):
        if not passage_hash in [item['hash'] for item in basket.all()]:
          basket.insert({
            'hash': passage_hash,
            'text': passage['text'],
            'source': passage['source'],
          })
        pass
      else:
        st.write(passage['text'])
        st.write(passage['source'])
        st.write(' ')
        if passage_hash in [item['hash'] for item in basket.all()]:
          basket.remove(
            doc_ids=[
              basket.get(Query().hash == passage_hash).doc_id
            ]
          )
    number_in_basket.text(f'Number of passages selected: {len(basket)}')

elif option == 'Passages selected':
  st.header("The passages you have selected")
  new_text = st.text_area('Enter your own passage:', value='', height=None)
  if new_text != '':
    basket.insert({
      'hash': hash(new_text),
      'text': new_text,
      'source': '',
    })
    rerun()
  st.markdown('*****')
  for passage in basket.all():
    st.write(passage['text'])
    st.write(passage['source'])
    if st.button('Remove', key='remove{}'.format(passage['hash'])):
        basket.remove(
          doc_ids=[
            basket.get(Query().hash == passage['hash']).doc_id
          ]
        )
        rerun()
    st.write('')

elif option == 'Summarization':
  st.header("Summarization of selected passages to create an article")
  title_of_article = st.text_input('Title of the article:', 'My article')
  st.markdown('*****')
  min_length = st.slider('Minimum length of the summarize:', 10, 200, 50)
  max_length = st.slider('Maximum length of the summarize:', 100, 400, 100)
  do_summarize = st.button('Summarize')
  st.markdown('*****')
  if do_summarize:
    document = ' '.join([passage['text'] for passage in basket.all()])
    summary = summarize(summarizer, document, max_length, min_length)
    st.title(title_of_article)
    st.write(summary)