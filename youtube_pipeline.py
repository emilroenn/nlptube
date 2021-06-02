import pandas as pd
import streamlit as st
import SessionState
from scraper_fns import *
from nlp_fns import * 
import base64

st.title('Fucking Fancy Youtube Scraper (TM)')

user_input1 = st.text_input("Search term:", '')
user_input2 = st.number_input("Number of results:", 1)

q_button = st.button('Scrape Youtube')

word_cloud_button = st.button('Create Wordcloud')

data_load_state = st.text('Loading script, please wait...')
#scrape_state = st.text(('Total videos scraped: ' + str(total_scraped)+". Total scrapes remaining: approximately " + str(10000-total_scraped)))

from nltk.corpus import stopwords
import gensim
from gensim.utils import simple_preprocess
import spacy
import re
import os #
import pickle
import pandas as pd
from pandas import DataFrame
import google.oauth2.credentials
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
from youtube_transcript_api import YouTubeTranscriptApi
from sklearn.feature_extraction.text import TfidfVectorizer
import matplotlib.pyplot as plt


from stqdm import stqdm
from wordcloud import WordCloud
from PIL import Image


def pipeline2 (df):

    progress = st.text('2/4: Running Term-Frequency Inverse-Document-Frequency analysis...')
    df = tfidf_prep(df)
    progress.text('3/4: Vectorizing documents...')
    df = vectorize(df)
    progress.text('4/4: Building wordcloud visualization...')
    wordcloud = visualize(df)
    return wordcloud

data_load_state.text('Script loaded! Awaiting input...')

ss = SessionState.get()

if q_button:
    ss.x = get_data(user_input1, user_input2)

######################### Download csv of scraped data ###################

if hasattr(ss, 'x'):
    csv = ss.x.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()  # some strings <-> bytes conversions necessary here
    href = f'<a href="data:file/csv;base64,{b64}">Download CSV File</a> (right-click and save as &lt;some_name&gt;.csv)'
    st.markdown(href, unsafe_allow_html=True)

############################################################################

if word_cloud_button:
    wordcloud = pipeline2(ss.x)
    st.image(wordcloud.to_array())

if st.button("test"):
    st.text("this fucking works")








#scrape_state.text(('Total videos scraped: ' + str(total_scraped)+". Total scrapes remaining: approximately " + str(10000-total_scraped)))


#################### SAVE FOR LATER MAYBE ########################

# @st.cache(allow_output_mutation=True,persist=True, suppress_st_warning=True)
# def pipeline1 (query,results):
#     data_load_state.text('1/4: Scraping YouTube...')
#     df = get_data(query, results)
#     data_load_state.text('2/4: Running Term-Frequency Inverse-Document-Frequency analysis...')
#     df = tfidf_prep(df)
#     data_load_state.text('3/4: Vectorizing documents...')
#     df = vectorize(df)
#     return df