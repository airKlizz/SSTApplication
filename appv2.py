import streamlit as st
from st_utils import rerun

from transformers import AutoTokenizer, AutoModelWithLMHead

from tinydb import TinyDB, Query
from newsplease import NewsPlease

@st.cache(allow_output_mutation=True)
def load_summarizer():
  tokenizer = AutoTokenizer.from_pretrained("airKlizz/bart-large-multi-en-wiki-news")
  model = AutoModelWithLMHead.from_pretrained("airKlizz/bart-large-multi-en-wiki-news").cuda()
  return {'model': model, 'tokenizer': tokenizer}

def summarize(summarizer, document, max_length, min_length):
  tokenizer = summarizer['tokenizer']
  model = summarizer['model']
  inputs = tokenizer.encode(document, return_tensors="pt", max_length=1024)
  outputs = model.generate(inputs.cuda(), max_length=max_length, min_length=min_length, length_penalty=2.0, num_beams=4, early_stopping=True)
  return tokenizer.decode(outputs[0], skip_special_tokens=True, clean_up_tokenization_spaces=False)

summarizer = load_summarizer()

st.sidebar.title('Semantic Storytelling Application')
st.sidebar.header('Multi-document summarization')

option = st.sidebar.selectbox('Menu:',
                      ('Text to summarize', 'Summarization')
                    )

db = TinyDB('documents.json').table('documents')

if option == 'Text to summarize':
    st.header('Enter an URL or a document to summarize')
    st.markdown('*****')
    new_url = st.text_input('Enter your URL:', value='')
    new_text = st.text_area('Enter your document:', value='')
    if new_url != '':
        article = NewsPlease.from_url(new_url)
        new_text = article.main_text
        db.insert({
            'hash': hash(new_text),
            'text': new_text,
        })
        rerun()
    elif new_text != '':
        db.insert({
            'hash': hash(new_text),
            'text': new_text,
        })
        rerun()
    st.markdown('*****')
    for passage in db.all():
        st.write(passage['text'])
        if st.button('Remove', key='remove{}'.format(passage['hash'])):
            db.remove(
                doc_ids=[
                    db.get(Query().hash == passage['hash']).doc_id
                ]
            )
            rerun()
        st.write('')

elif option == 'Summarization':
    st.header("Summarization of selected passages to create an article")
    st.markdown('*****')
    st.write('Text to summarize:')
    document = ' '.join([passage['text'] for passage in db.all()][:1024])
    st.write(document)
    st.markdown('*****')
    min_length = st.slider('Minimum length of the summarize:', 100, 300, 200)
    max_length = st.slider('Maximum length of the summarize:', 300, 500, 400)
    do_summarize = st.button('Summarize')
    st.markdown('*****')
    if do_summarize:
        text_to_summarize = ' '.join([passage['text'] for passage in db.all()])
        summary = summarize(summarizer, text_to_summarize, max_length, min_length)
        st.write(summary)


