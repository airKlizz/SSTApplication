import streamlit as st

import tensorflow as tf
import numpy as np
from transformers import TFAutoModel, AutoTokenizer
from MsMarco.bonus.search_engine import *
from MsMarco.model.scorer import Scorer

_MAX_LENGTH = 256
_NUM_CLASSES = 2
_MODEL_NAME = 'roberta-large'
_WEIGTH_PATH = 'MsMarco/model/saved_weights/model_roberta-large_mrr_0.303.h5'

tokenizer = AutoTokenizer.from_pretrained(_MODEL_NAME)
model = Scorer(tokenizer, TFAutoModel, _MAX_LENGTH, _NUM_CLASSES)
model.from_pretrained(_MODEL_NAME)
model(tf.zeros([1, 3, _MAX_LENGTH], tf.int32))
model.load_weights(_WEIGTH_PATH)
model.compile(run_eagerly=True)

st.title('Semantic Storytelling Application')
st.header('Create an news article automaticly')
title = st.text_input('Enter the title/topic of your article:', '')

if title != '':
    st.write(f'Let\'s write an article about: {title}')