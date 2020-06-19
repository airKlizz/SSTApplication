import streamlit as st

import torch
from transformers import AutoTokenizer, AutoModelWithLMHead
from nlp import load_dataset
import random

# Utils

@st.cache(allow_output_mutation=True)
def load_tokenizer_and_model(tokenizer_name, model_name):
  tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
  model = AutoModelWithLMHead.from_pretrained(model_name)
  if torch.cuda.is_available():
      model.cuda()
  return tokenizer, model

def summarize(tokenizer, model, document, max_length, min_length):
    inputs = tokenizer.encode(document, return_tensors="pt", pad_to_max_length=True, max_length=tokenizer.max_len)
    if torch.cuda.is_available():
        outputs = model.generate(inputs.cuda(), max_length=max_length, min_length=min_length, length_penalty=2.0, num_beams=4, early_stopping=True)
    else:
        outputs = model.generate(inputs, max_length=max_length, min_length=min_length, length_penalty=2.0, num_beams=4, early_stopping=True)
    return tokenizer.decode(outputs[0], skip_special_tokens=True, clean_up_tokenization_spaces=False)

en_dataset = load_dataset('en_wiki_multi_news.py', cache_dir='dataset/.en-wiki-multi-news-cache', split='test')
de_dataset = load_dataset('de_wiki_multi_news.py', cache_dir='dataset/.de-wiki-multi-news-cache', split='test')
fr_dataset = load_dataset('fr_wiki_multi_news.py', cache_dir='dataset/.fr-wiki-multi-news-cache', split='test')


# Part 0: Title

st.markdown(
    """
    # Generate long summaries in multiple languages using Transformers

    We use BART or T5 to generate long summaries in English, German or French. This application is a demo of a research paper.

    *******
    """
)

# Part 1: Chose the text to summarize
st.markdown("## Text to summarize")

seed = st.number_input("Insert the random seed:", value=42)
random.seed(seed)

text_selected = st.selectbox(
    "Select 'Custom text' or an example (randomly chosen in Multi-Wiki-News):",
    ("Custom text", "English example", "German example", "French example")
)

text_to_summarize = ""
if text_selected == "Custom text":
    text_to_summarize = st.text_area("Enter the text to summarize:", height=400)
else:
    if text_selected == "English example":
        text_to_summarize = random.choice(en_dataset['document'])
    elif text_selected == "German example":
        text_to_summarize = random.choice(de_dataset['document'])
    elif text_selected == "French example":
        text_to_summarize = random.choice(fr_dataset['document'])
    text_to_summarize = st.text_area("Text to summarize:", text_to_summarize, height=400)

st.markdown("******")

# Part 2: Chose parameters
st.markdown("## Parameters")

model = st.selectbox(
    "Select a model:",
    (
        "bart-large-cnn-multi-en-wiki-news",
        "bart-large-multi-combine-wiki-news",
        "bart-large-multi-de-wiki-news",
        "bart-large-multi-en-wiki-news",
        "bart-large-multi-fr-wiki-news",
        "bert2bert-multi-de-wiki-news",
        "bert2bert-multi-en-wiki-news",
        "bert2bert-multi-fr-wiki-news",
        "t5-base-multi-combine-wiki-news",
        "t5-base-multi-de-wiki-news",
        "t5-base-multi-en-wiki-news",
        "t5-base-multi-fr-wiki-news",
        "t5-base-with-title-multi-de-wiki-news",
        "t5-base-with-title-multi-en-wiki-news",
        "t5-base-with-title-multi-fr-wiki-news",
    )
)

if "bart" in model:
    tokenizer_name = "facebook/bart-large"
elif "bert2bert" in model:
    tokenizer_name = "bert-base-cased"
elif "t5" in model:
    tokenizer_name = "t5-base"
    if "with-title" in model:
        title = st.text_input("Title of the summary:")
        task_prefix = st.selectbox(
            "Select a task prefix:",
            (f"title: {title}  summarize:", f"titel: {title}  zusammenfassen:", f"title: {title}  résume:")
        )
    else:
        task_prefix = st.selectbox(
            "Select a task prefix:",
            ("summarize:", "zusammenfassen:", "résume:")
        )
    text_to_summarize = task_prefix + " " + text_to_summarize

model_name = "airKlizz/" + model

tokenizer, model = None, None
if st.button('Load'):
    tokenizer, model = load_tokenizer_and_model(tokenizer_name, model_name)

min_length = st.slider('Minimum length of the summarize:', 100, 300, 200)
max_length = st.slider('Maximum length of the summarize:', 300, 500, 400)

st.markdown("******")

# Part 3: Summarization
st.markdown("## Summarization")

#st.markdown("**Document: **" + text_to_summarize)

if st.button('Run'):
    if text_to_summarize == "":
        st.write("Please enter a text to summarize or chose an example.")
    else:
        if tokenizer == None and model == None:
            tokenizer, model = load_tokenizer_and_model(tokenizer_name, model_name)
        summary = summarize(tokenizer, model, text_to_summarize, max_length, min_length)
        st.markdown("**Summary: **" + summary)