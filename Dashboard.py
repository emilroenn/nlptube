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

from enum import Enum
from io import BytesIO, StringIO
from typing import Union

import pandas as pd
import streamlit as st




#### POSSIBLE FUNCTIONS TO MOVE ####

def pipeline_multiple (df_list, cloud_color, cloud_bg, cloud_shape, cloud_font):
    progress = st.text('1/3: Running Term-Frequency Inverse-Document-Frequency analysis...')
    total_dfs = len(df_list)
    df_list = tfidf_prep(df_list)
    progress.text('2/3: Vectorizing documents...')
    df = vectorize_multiple(df_list)
    progress.text('3/3: Building wordcloud visualization...')
    wordcloud_list = []
    for x in range(total_dfs):
        selected_df = df[[x]]
        wordcloud = visualize(selected_df, cloud_color, cloud_bg, cloud_shape, cloud_font, x)
        wordcloud_list.append(wordcloud)
    progress.text('Done!')
    return wordcloud_list

def pipeline_single (df, cloud_color, cloud_bg, cloud_shape, cloud_font):
    progress = st.text('1/2: Vectorizing documents...')
 #   df = tfidf_prep(df)
  #  progress.text('3/4: Vectorizing documents...')
    df = vectorize_single(df)
    progress.text('2/2: Building wordcloud visualization...')
    wordcloud = visualize(df, cloud_color, cloud_bg, cloud_shape, cloud_font)
    progress.text('Done!')
    return wordcloud



ss = SessionState.get(current = "Welcome", 
                data1q = "NA", data1s = 0, data1tf = "NA",
                data2q = "NA", data2s = 0, data2tf = "NA", 
                data3q = "NA", data3s = 0, data3tf = "NA")
#ss.ticker = 0
#ss.x = 
#ss.current = "Main"


def main():

    # Dictionary of pages
    pages = {
        "Welcome": page_main,
        "How it works": page_how,
        "Upload Data": page_upload,
        "Scrape YouTube Data": page_scrape,
        "Pre-process Data": page_pre,
        "Visualize Data" : page_visualize,
        "About Us" : page_about
    }

    # Sidebar title
    st.sidebar.title("Navigation")
    st.sidebar.header("")

    # Sidebar buttons - if a button is True, state will be set to the page of that button
    if st.sidebar.button("Welcome"):
        ss.current = "Welcome"
    if st.sidebar.button("How it works"):
        ss.current = "How it works"
    if st.sidebar.button("Upload Data"):
        ss.current = "Upload Data"
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

def page_upload():
    st.title('Upload your own motherfucking data')
    st.info("Select the data container that you wish to upload your data to")
    dataset = st.selectbox("",['Data 1', 'Data 2', 'Data 3'])

    STYLE = """
    <style>
    img {
        max-width: 100%;
    }
    </style>
    """

    FILE_TYPES = ["csv", "py", "png", "jpg"]


    class FileType(Enum):
        """Used to distinguish between file types"""

        IMAGE = "Image"
        CSV = "csv"
        PYTHON = "Python"

    def get_file_type(file: Union[BytesIO, StringIO]) -> FileType:
        """The file uploader widget does not provide information on the type of file uploaded so we have
        to guess using rules or ML. See
        [Issue 896](https://github.com/streamlit/streamlit/issues/896)

        I've implemented rules for now :-)

        Arguments:
            file {Union[BytesIO, StringIO]} -- The file uploaded

        Returns:
            FileType -- A best guess of the file type
        """

        return FileType.CSV

    def main2():
        """Run this function to display the Streamlit app"""
        st.markdown(STYLE, unsafe_allow_html=True)

        file = st.file_uploader("Upload file", type="CSV")
        show_file = st.empty()
        if not file:
            show_file.info("Please upload a file of type: CSV " )
            return

        file_type = get_file_type(file)
        # if file_type == FileType.IMAGE:
            # st.info("please upload a .csv file in the scraper format, and not an image")
        # elif file_type == FileType.PYTHON:
            # st.info("please upload a .csv file in the scraper format, and not and not python code")
        if file_type == FileType.CSV:
            if dataset == "Data 1":
                data = pd.read_csv(file)
                ss.data1 = data
                ss.data1q = data.at[2,'Query']
                ss.data1s = len(ss.data1)
                csv = ss.data1.to_csv(index=False)
                b64 = base64.b64encode(csv.encode()).decode()  # some strings <-> bytes conversions necessary here
                ss.data1_href = f'<a href="data:file/csv;base64,{b64}">Download CSV File</a> (right-click and save as &lt;some_name&gt;.csv)'
                st.info(f"file using the query '{ss.data1q}' with {ss.data1s} comments successfully uploaded to data container 1")
                st.dataframe(data.head(10))
            if dataset == "Data 2":
                data = pd.read_csv(file)
                ss.data2 = data
                ss.data2q = data.at[2,'Query']
                ss.data2s = len(ss.data2)
                csv = ss.data2.to_csv(index=False)
                b64 = base64.b64encode(csv.encode()).decode()  # some strings <-> bytes conversions necessary here
                ss.data2_href = f'<a href="data:file/csv;base64,{b64}">Download CSV File</a> (right-click and save as &lt;some_name&gt;.csv)'
                st.info(f"file using the query '{ss.data2q}' with {ss.data2s} comments successfully uploaded to data container 2")
                st.dataframe(data.head(10))
            if dataset == "Data 3":
                data = pd.read_csv(file)
                ss.data3 = data
                ss.data3q = data.at[2,'Query']
                ss.data3s = len(ss.data3)
                csv = ss.data3.to_csv(index=False)
                b64 = base64.b64encode(csv.encode()).decode()  # some strings <-> bytes conversions necessary here
                ss.data3_href = f'<a href="data:file/csv;base64,{b64}">Download CSV File</a> (right-click and save as &lt;some_name&gt;.csv)'
                st.info(f"file using the query '{ss.data3q}' with {ss.data3s} comments successfully uploaded to data container 3")
                st.dataframe(data.head(10))

        file.close()
        
    main2()

    
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



def page_pre():
    st.title('PREPROCESS, MOTHERFUCKER. DO YOU SPEAK IT?')
    st.text("Here's where we help you preprocess your data.")   

def page_visualize():
    st.title('LETS GET SOME VISUALS UP IN HERE')
    st.text("This is where we visualize shit")   

    type = st.selectbox("Analysis:", ['Individual datasets', 'Differences between datasets'])

    if type == "Individual datasets":
        with st.form(key='my_form'):
            dataset = st.selectbox("Select a dataset to visualize:", ['Data 1', 'Data 2', 'Data 3'])
            st.text("Customize wordcloud:")   
            col1, col2 = st.beta_columns(2)
            cloud_color = col1.selectbox("Theme:", ['Default (Black)','summer', 'Wistia', 'OrRd', 'YlGn'])
            cloud_font = col1.selectbox("Font:", ['Default (AU Passata)','AU', 'SpicyRice'])
            cloud_bg = col2.selectbox("Background:", ['Default (White)','black', 'white', 'red'])
            cloud_shape = col2.selectbox("Shape:", ['Default (Square)','Circle', 'Heart'])
            submit_button = st.form_submit_button(label='Create Wordcloud')

        try:
            if submit_button:
                if dataset == "Data 1":
                    wordcloud = pipeline_single(ss.data1, cloud_color, cloud_bg, cloud_shape, cloud_font)
                if dataset == "Data 2":
                    wordcloud = pipeline_single(ss.data2, cloud_color, cloud_bg, cloud_shape, cloud_font)
                if dataset == "Data 3":
                    wordcloud = pipeline_single(ss.data3, cloud_color, cloud_bg, cloud_shape, cloud_font)
                st.image(wordcloud.to_array())
        except AttributeError as e:
            st.write("Please scrape some data first, fool!")
            st.write("Error:", e)
        
    
    if type == "Differences between datasets":
        with st.form(key='my_form'):
        #    dataset = st.selectbox("Select a dataset to visualize:", ['Data 1', 'Data 2', 'Data 3'])
            st.text("Customize wordcloud:")   
            col1, col2 = st.beta_columns(2)
            cloud_color = col1.selectbox("Theme:", ['Default (Black)','summer', 'Wistia', 'OrRd', 'YlGn'])
            cloud_font = col1.selectbox("Font:", ['Default (AU Passata)','AU', 'SpicyRice'])
            cloud_bg = col2.selectbox("Background:", ['Default (White)','black', 'white', 'red'])
            cloud_shape = col2.selectbox("Shape:", ['Default (Square)','Circle', 'Heart'])
            submit_button = st.form_submit_button(label='Create Wordcloud')

        try:
            if submit_button:
                df_list = []
                if hasattr(ss, 'data1'):
                    df_list.append(ss.data1)
                if hasattr(ss, 'data2'):
                    df_list.append(ss.data2)
                if hasattr(ss, 'data3'):
                    df_list.append(ss.data3)
                wordcloud_list = pipeline_multiple(df_list, cloud_color, cloud_bg, cloud_shape, cloud_font)
                

              #  with st.form(key='my_form'):
              #      st.text("Customize wordcloud:")   
              #      col1, col2 = st.beta_columns(2)
                st.image(wordcloud_list[0].to_array())
                st.image(wordcloud_list[1].to_array())
             #   st.image(wordcloud_list[2].to_array())

        except AttributeError as e:
            st.write("Please scrape some data first, fool!")
            st.write("Error:", e)
        
        

def page_about():
    st.title('Fucking Fancy Youtube Scraper (TM)')
    st.text("Good for you, motherfucker")   


if __name__ == "__main__":
    main()



