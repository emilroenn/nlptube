import streamlit as st
import SessionState
from scraper_fns import *
from nlp_fns import *
import base64

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




#### POSSIBLE FUNCTIONS TO MOVE ####

def pipeline2 (df):
    progress = st.text('2/4: Running Term-Frequency Inverse-Document-Frequency analysis...')
    df = tfidf_prep(df)
    progress.text('3/4: Vectorizing documents...')
    df = vectorize(df)
    progress.text('4/4: Building wordcloud visualization...')
    wordcloud = visualize(df)
    return wordcloud



ss = SessionState.get(current = "Welcome", data1q = "NA", data1s = 0, data2q = "NA", data2s = 0, data3q = "NA", data3s = 0)
#ss.ticker = 0
#ss.x = 
#ss.current = "Main"


def main():

    # Dictionary of pages
    pages = {
        "Welcome": page_main,
        "How it works": page_how,
        "Scrape YouTube Data": page_scrape,
        "Pre-process Data": page_pre,
        "Visualize Data" : page_visualize,
        "About us" : page_about
    }

    # Sidebar title
    st.sidebar.title("Navigation")
    st.sidebar.header("")

    # Sidebar buttons - if a button is True, state will be set to the page of that button
    if st.sidebar.button("Welcome"):
        ss.current = "Welcome"
    if st.sidebar.button("How it works"):
        ss.current = "How it works"
    if st.sidebar.button("Scrape YouTube Data"):
        ss.current = "Scrape YouTube Data"
    if st.sidebar.button("Pre-process Data"):
        ss.current = "Pre-process Data"
    if st.sidebar.button("Visualize Data"):
        ss.current = "Visualize Data"
    if st.sidebar.button("About Us"):
        ss.current = "About Us"

    st.sidebar.header(":floppy_disk: Data Stored:")
    if ss.data1q == "NA":
        data1text = "DATA 1: Empty"
    if ss.data1q != "NA":
        data1text = "DATA 1: " + str(ss.data1q) + " (" + str(ss.data1s) + " comments)"
    if ss.data2q == "NA":
        data2text = "DATA 2: Empty"
    if ss.data2q != "NA":
        data2text = "DATA 2: " + str(ss.data2q) + " (" + str(ss.data2s) + " comments)"
    if ss.data3q == "NA":
        data3text = "DATA 3: Empty"
    if ss.data3q != "NA":
        data3text = "DATA 3: " + str(ss.data3q) + " (" + str(ss.data3s) + " comments)"
    st.sidebar.text(data1text)
    if hasattr(ss, 'data1_href'):
        st.sidebar.markdown(ss.data1_href, unsafe_allow_html=True)
    st.sidebar.text(data2text)
    if hasattr(ss, 'data2_href'):
        st.sidebar.markdown(ss.data2_href, unsafe_allow_html=True)
    st.sidebar.text(data3text)
    if hasattr(ss, 'data3_href'):
        st.sidebar.markdown(ss.data3_href, unsafe_allow_html=True)


    # Display the selected page with the session state
    pages[ss.current]()
    


def page_main():
    st.title('Welcome to the Fucking Fancy Youtube Scraper (TM)')
    st.header("This is where we put a motherfucking header")
    st.write("This is where the welcome text fucking goes, motherfucker.")

def page_how():
    st.title('How to use this tool')
    st.write("This is where we tell you how to use it, motherfucker")

def page_scrape():
    st.title('SCRAPE SOME FUCKING DATA YO')
    st.text("Let's get you some fucking data. You can store up to three datasets for analysis and comparison.")
    dataset = st.selectbox("Select a dataset to create", ['Data 1', 'Data 2', 'Data 3'])
    user_input1 = st.text_input("Search term:", '')
    user_input2 = st.number_input("Number of results:", 1)

    start_button = st.button('Scrape Youtube')

    if start_button:
        progress = st.header("Scraping YouTube, please wait...")
        if dataset == "Data 1":
            ss.data1 = get_data(user_input1, user_input2)
            ss.data1q = user_input1
            ss.data1s = len(ss.data1)
            total = len(ss.data1)
            csv = ss.data1.to_csv(index=False)
            b64 = base64.b64encode(csv.encode()).decode()  # some strings <-> bytes conversions necessary here
            ss.data1_href = f'<a href="data:file/csv;base64,{b64}">Download CSV File</a> (right-click and save as &lt;some_name&gt;.csv)'
            st.markdown(ss.data1_href, unsafe_allow_html=True)
        if dataset == "Data 2":
            ss.data2 = get_data(user_input1, user_input2)
            ss.data2q = user_input1
            ss.data2s = len(ss.data2)
            total = len(ss.data2)
            csv = ss.data2.to_csv(index=False)
            b64 = base64.b64encode(csv.encode()).decode()  # some strings <-> bytes conversions necessary here
            ss.data2_href = f'<a href="data:file/csv;base64,{b64}">Download CSV File</a> (right-click and save as &lt;some_name&gt;.csv)'
            st.markdown(ss.data2_href, unsafe_allow_html=True)
        if dataset == "Data 3":
            ss.data3 = get_data(user_input1, user_input2)
            ss.data3q = user_input1
            ss.data3s = len(ss.data3)
            total = len(ss.data3)
            csv = ss.data3.to_csv(index=False)
            b64 = base64.b64encode(csv.encode()).decode()  # some strings <-> bytes conversions necessary here
            ss.data3_href = f'<a href="data:file/csv;base64,{b64}">Download CSV File</a> (right-click and save as &lt;some_name&gt;.csv)'
            st.markdown(ss.data3_href, unsafe_allow_html=True)
        progress.header("Done!")
        st.write("Total comments scraped:", total)


        
  #  download_button = st.button('Download data')
  #  if download_button:
  #      st.write("This is where we download shit")



def page_pre():
    st.title('PREPROCESS, MOTHERFUCKER. DO YOU SPEAK IT?')
    st.text("Here's where we help you preprocess your data.")   

def page_visualize():
    st.title('LETS GET SOME VISUALS UP IN HERE')
    st.text("This is where we visualize shit")   
    dataset = st.selectbox("Select a dataset to visualize:", ['Data 1', 'Data 2', 'Data 3'])

    word_cloud_button = st.button('Create Wordcloud')

    try:
        if word_cloud_button:
            if dataset == "Data 1":
                wordcloud = pipeline2(ss.data1)
            if dataset == "Data 2":
                wordcloud = pipeline2(ss.data2)
            if dataset == "Data 3":
                wordcloud = pipeline2(ss.data3)
            st.image(wordcloud.to_array())
    except AttributeError:
        st.write("Please scrape some data first, fool!")

def page_about():
    st.title('Fucking Fancy Youtube Scraper (TM)')
    st.text("Good for you, motherfucker")   
    back = st.button("Get back") 

    if back:
        ss.current = "Main"

if __name__ == "__main__":
    main()



