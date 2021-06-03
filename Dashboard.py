import streamlit as st
import SessionState
from scraper_fns import *
from nlp_fns import *
import base64
import altair as alt
import time

from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk
nltk.download('vader_lexicon')
import numpy as np
sid = SentimentIntensityAnalyzer()

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

ss = SessionState.get(current = "Welcome", upload = "NA", wc = "NA", df_list = [1,1,1,1,1], query_list = [1,1,1,1,1],
                data1q = "NA", data1s = 0, data1tf = "NA",
                data2q = "NA", data2s = 0, data2tf = "NA", 
                data3q = "NA", data3s = 0, data3tf = "NA",
                data4q = "NA", data4s = 0, data4tf = "NA",
                data5q = "NA", data5s = 0, data5tf = "NA")

def main():


    st.markdown(
            f"""
    <style>
        .reportview-container .main .block-container{{
            max-width: {2000}px;
            padding-top: {2}rem;
            padding-right: {5}rem;
            padding-left: {5}rem;
            padding-bottom: {5}rem;
        }}

    </style>
    """,
            unsafe_allow_html=True,
        )


    # Dictionary of pages
    pages = {
        "Welcome": page_main,
        "How it works": page_how,
        "Upload Data": page_upload,
        "Scrape YouTube Data": page_scrape,
        "Wordcloud Analysis" : page_visualize,
        "Sentiment Analysis" : page_sentiment,
        "Topic Analysis" : page_topic,
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
    if st.sidebar.button("Wordcloud Analysis"):
        ss.current = "Wordcloud Analysis"
    if st.sidebar.button("Sentiment Analysis"):
        ss.current = "Sentiment Analysis"
    if st.sidebar.button("Topic Analysis"):
        ss.current = "Topic Analysis"
    if st.sidebar.button("About Us"):
        ss.current = "About Us"

    
    st.sidebar.header(":floppy_disk: Data Stored:")
    if ss.data1q == "NA":
        data1text = "CONTAINER 1: Empty"
    if ss.data1q != "NA":
        data1text = "CONTAINER 1: " + str(ss.data1q) + " (" + str(ss.data1s) + " comments)"
    if ss.data2q == "NA":
        data2text = "CONTAINER 2: Empty"
    if ss.data2q != "NA":
        data2text = "CONTAINER 2: " + str(ss.data2q) + " (" + str(ss.data2s) + " comments)"
    if ss.data3q == "NA":
        data3text = "CONTAINER 3: Empty"
    if ss.data3q != "NA":
        data3text = "CONTAINER 3: " + str(ss.data3q) + " (" + str(ss.data3s) + " comments)"
    if ss.data4q == "NA":
        data4text = "CONTAINER 4: Empty"
    if ss.data4q != "NA":
        data4text = "CONTAINER 4: " + str(ss.data4q) + " (" + str(ss.data4s) + " comments)"
    if ss.data5q == "NA":
        data5text = "CONTAINER 5: Empty"
    if ss.data5q != "NA":
        data5text = "CONTAINER 5: " + str(ss.data5q) + " (" + str(ss.data5s) + " comments)"

    st.sidebar.text(data1text)
    if hasattr(ss, 'data1_href'):
        st.sidebar.markdown(ss.data1_href, unsafe_allow_html=True)
    st.sidebar.text(data2text)
    if hasattr(ss, 'data2_href'):
        st.sidebar.markdown(ss.data2_href, unsafe_allow_html=True)
    st.sidebar.text(data3text)
    if hasattr(ss, 'data3_href'):
        st.sidebar.markdown(ss.data3_href, unsafe_allow_html=True)
    st.sidebar.text(data4text)
    if hasattr(ss, 'data4_href'):
        st.sidebar.markdown(ss.data4_href, unsafe_allow_html=True)
    st.sidebar.text(data5text)
    if hasattr(ss, 'data5_href'):
        st.sidebar.markdown(ss.data5_href, unsafe_allow_html=True)


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
    
    col1, col2, col3, col4, col5 = st.beta_columns(5)
    container1 = col1.button("Data Container 1")
    container2 = col2.button("Data Container 2")
    container3 = col3.button("Data Container 3")
    container4 = col4.button("Data Container 4")
    container5 = col5.button("Data Container 5")
  #  dataset = st.selectbox("",['Data 1', 'Data 2', 'Data 3'])
    
    if container1:
        ss.upload = "Data 1"
    if container2:
        ss.upload = "Data 2"
    if container3:
        ss.upload = "Data 3"
    if container4:
        ss.upload = "Data 3"
    if container5:
        ss.upload = "Data 3"

    dataset = ss.upload
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
                ss.df_list[0] = ss.data1
                ss.query_list[0] = ss.data1q
                csv = ss.data1.to_csv(index=False)
                b64 = base64.b64encode(csv.encode()).decode()  # some strings <-> bytes conversions necessary here
                ss.data1_href = f'<a href="data:file/csv;base64,{b64}" download="{ss.data1q}.csv">Download CSV</a>'
                st.info(f"file using the query '{ss.data1q}' with {ss.data1s} comments successfully uploaded to data container 1")
                st.dataframe(data.head(10))
            if dataset == "Data 2":
                data = pd.read_csv(file)
                ss.data2 = data
                ss.data2q = data.at[2,'Query']
                ss.data2s = len(ss.data2)
                ss.df_list[1] = ss.data2
                ss.query_list[1] = ss.data2q
                csv = ss.data2.to_csv(index=False)
                b64 = base64.b64encode(csv.encode()).decode()  # some strings <-> bytes conversions necessary here
                ss.data2_href = f'<a href="data:file/csv;base64,{b64}" download="{ss.data2q}.csv">Download CSV</a>'
                st.info(f"file using the query '{ss.data2q}' with {ss.data2s} comments successfully uploaded to data container 2")
                st.dataframe(data.head(10))
            if dataset == "Data 3":
                data = pd.read_csv(file)
                ss.data3 = data
                ss.data3q = data.at[2,'Query']
                ss.data3s = len(ss.data3)
                ss.df_list[2] = ss.data3
                ss.query_list[2] = ss.data3q
                csv = ss.data3.to_csv(index=False)
                b64 = base64.b64encode(csv.encode()).decode()  # some strings <-> bytes conversions necessary here
                ss.data3_href = f'<a href="data:file/csv;base64,{b64}" download="{ss.data3q}.csv">Download CSV</a>'
                st.info(f"file using the query '{ss.data3q}' with {ss.data3s} comments successfully uploaded to data container 3")
                st.dataframe(data.head(10))
            if dataset == "Data 4":
                data = pd.read_csv(file)
                ss.data4 = data
                ss.data4q = data.at[2,'Query']
                ss.data4s = len(ss.data4)
                ss.df_list[3] = ss.data4
                ss.query_list[3] = ss.data4q
                csv = ss.data4.to_csv(index=False)
                b64 = base64.b64encode(csv.encode()).decode()  # some strings <-> bytes conversions necessary here
                ss.data4_href = f'<a href="data:file/csv;base64,{b64}" download="{ss.data4q}.csv">Download CSV</a>'
                st.info(f"file using the query '{ss.data4q}' with {ss.data4s} comments successfully uploaded to data container 3")
                st.dataframe(data.head(10))
            if dataset == "Data 5":
                data = pd.read_csv(file)
                ss.data5 = data
                ss.data5q = data.at[2,'Query']
                ss.data5s = len(ss.data5)
                ss.df_list[4] = ss.data5
                ss.query_list[4] = ss.data5q
                csv = ss.data5.to_csv(index=False)
                b64 = base64.b64encode(csv.encode()).decode()  # some strings <-> bytes conversions necessary here
                ss.data5_href = f'<a href="data:file/csv;base64,{b64}" download="{ss.data5q}.csv">Download CSV</a>'
                st.info(f"file using the query '{ss.data5q}' with {ss.data5s} comments successfully uploaded to data container 3")
                st.dataframe(data.head(10))

        file.close()
        
    main2()

    
def page_scrape():
    st.header('YouTube Scraper')
    with st.form(key='my_form'):
        col1, col2, col3 = st.beta_columns([1,2,3])
        col3.write("**To analyse data, you gotta have data first!**  \n This tool executes a YouTube search for an input query, finds the number of videos selected, and scrapes up to the top 100 comments of each video. Detailed information about both the videos and comments are then converted to a data frame and stored in the selected data container in the app. After the scraping is complete, feel free to download the data frame or use one of the tools in the sidebar for further analysis!")
        dataset = col1.radio("Store data in:", ('Container 1', 'Container 2', 'Container 3', 'Container 4', 'Container 5'))
        user_input1 = col2.text_input("Search term:", '')
        user_input2 = col2.number_input("Videos:", 1)
        st.info("Please note: YouTube's API has a daily limit of 10,000 requests. Please limit searches to 100 videos or less.")
        submit_button = st.form_submit_button(label='Start Scraper')


  #  start_button = st.button('Scrape Youtube')


#    data_expander = st.beta_expander()
#    with my_expander:
#        'Hello there!'
#        clicked = st.button('Click me!')

    if submit_button:
        if user_input1 == "":
            st.info("Please enter a search term before scraping!")
        else:
            progress = st.header("Scraping YouTube, please wait...")
            if dataset == "Container 1":
                ss.data1 = get_data(user_input1, user_input2)
                ss.data1q = user_input1
                ss.data1s = len(ss.data1)
                ss.df_list[0] = ss.data1
                ss.query_list[0] = ss.data1q
                total = len(ss.data1)
                csv = ss.data1.to_csv(index=False)
                b64 = base64.b64encode(csv.encode()).decode()  # some strings <-> bytes conversions necessary here
                ss.data1_href = f'<a href="data:file/csv;base64,{b64}" download="{ss.data1q}.csv">Click here to download CSV</a>'
                st.markdown(ss.data1_href, unsafe_allow_html=True)
                st.dataframe(ss.data1.head(10))
            if dataset == "Container 2":
                ss.data2 = get_data(user_input1, user_input2)
                ss.data2q = user_input1
                ss.data2s = len(ss.data2)
                ss.df_list[1] = ss.data2
                ss.query_list[1] = ss.data2q
                total = len(ss.data2)
                csv = ss.data2.to_csv(index=False)
                b64 = base64.b64encode(csv.encode()).decode()  # some strings <-> bytes conversions necessary here
                ss.data2_href = f'<a href="data:file/csv;base64,{b64}" download="{ss.data2q}.csv">Click here to download CSV</a>'
                st.markdown(ss.data2_href, unsafe_allow_html=True)
            if dataset == "Container 3":
                ss.data3 = get_data(user_input1, user_input2)
                ss.data3q = user_input1
                ss.data3s = len(ss.data3)
                ss.df_list[2] = ss.data3
                ss.query_list[2] = ss.data3q
                total = len(ss.data3)
                csv = ss.data3.to_csv(index=False)
                b64 = base64.b64encode(csv.encode()).decode()  # some strings <-> bytes conversions necessary here
                ss.data3_href = f'<a href="data:file/csv;base64,{b64}" download="{ss.data3q}.csv">Click here to download CSV</a>'
                st.markdown(ss.data3_href, unsafe_allow_html=True)
            if dataset == "Container 4":
                ss.data4 = get_data(user_input1, user_input2)
                ss.data4q = user_input1
                ss.data4s = len(ss.data4)
                ss.df_list[3] = ss.data4
                ss.query_list[3] = ss.data4q
                total = len(ss.data4)
                csv = ss.data4.to_csv(index=False)
                b64 = base64.b64encode(csv.encode()).decode()  # some strings <-> bytes conversions necessary here
                ss.data4_href = f'<a href="data:file/csv;base64,{b64}" download="{ss.data4q}.csv">Click here to download CSV</a>'
                st.markdown(ss.data4_href, unsafe_allow_html=True)
            if dataset == "Container 5":
                ss.data5 = get_data(user_input1, user_input2)
                ss.data5q = user_input1
                ss.data5s = len(ss.data5)
                ss.df_list[4] = ss.data5
                ss.query_list[4] = ss.data5q
                total = len(ss.data5)
                csv = ss.data5.to_csv(index=False)
                b64 = base64.b64encode(csv.encode()).decode()  # some strings <-> bytes conversions necessary here
                ss.data5_href = f'<a href="data:file/csv;base64,{b64}" download="{ss.data5q}.csv">Click here to download CSV</a>'
                st.markdown(ss.data5_href, unsafe_allow_html=True)
            progress.header("Done!")
            st.write("Total comments scraped:", total)



def page_visualize():
    st.header('**Wordclouds**')
    st.write("Wordclouds are one of the most simple yet effective visualizations of large amounts of text data. A tag cloud (word cloud or wordle or weighted list in visual design) is a novelty visual representation of text data, typically used to depict keyword metadata (tags) on websites, or to visualize free form text. Tags are usually single words, and the importance of each tag is shown with font size or color.[2] This format is useful for quickly perceiving the most prominent terms to determine its relative prominence. Bigger term means greater weight.")   

    type = st.selectbox("Analysis:", ['Individual datasets', 'Compare Datasets'])

    if type == "Individual datasets":

        col1, col2, col3, col4, col5, _ = st.beta_columns([1,1,1,1,1,2])
        container1 = col1.button("Dataset 1")
        container2 = col2.button("Dataset 2")
        container3 = col3.button("Dataset 3")
        container4 = col4.button("Dataset 4")
        container5 = col5.button("Dataset 5")
            
        if container1:
            ss.wc = "Data 1"
        if container2:
            ss.wc = "Data 2"
        if container3:
            ss.wc = "Data 3"
        if container4:
            ss.wc = "Data 3"
        if container5:
            ss.wc = "Data 3"

        dataset = ss.wc

        with st.form(key='my_form'):
            #dataset = st.selectbox("Select a dataset to visualize:", ['Data 1', 'Data 2', 'Data 3'])

            st.text("Customize wordcloud:")   
            col1, col2 = st.beta_columns(2)
            cloud_color = col1.selectbox("Theme:", ['Default (Black)','summer', 'Wistia', 'OrRd', 'YlGn'])
            cloud_font = col1.selectbox("Font:", ['Default (AU Passata)','AU', 'SpicyRice'])
            cloud_bg = col2.selectbox("Background:", ['Default (White)','black', 'white', 'red'])
            cloud_shape = col2.selectbox("Shape:", ['Default (Square)','Circle', 'Heart'])
            extra_stopwords = st.text_input("Remove stopwords (please separate words by comma):")
            submit_button = st.form_submit_button(label='Create Wordcloud')
        
        progress = st.header("Ready to plot! Click 'Create Wordcloud' to begin.")
        progressbar = st.progress(0)

        try:
            if submit_button:
                progress.header("Building wordcloud, please wait...")
                progressbar.progress(0.1)
                if dataset == "Data 1":
                    if hasattr(ss, "data1_prep"):
                        progressbar.progress(0.2)
                        wordcloud = vectorize_multiple([ss.data1_prep], extra_stopwords)
                        progressbar.progress(0.4)
                        wordcloud = visualize(wordcloud, cloud_color, cloud_bg, cloud_shape, cloud_font)
                        progressbar.progress(0.9)
                    else:
                        progressbar.progress(0.2)
                        ss.data1_prep = prep([ss.data1])
                        progressbar.progress(0.4)
                        wordcloud = vectorize_multiple([ss.data1_prep], extra_stopwords)
                        progressbar.progress(0.6)
                        wordcloud = visualize(wordcloud, cloud_color, cloud_bg, cloud_shape, cloud_font)
                        progressbar.progress(0.9)
                if dataset == "Data 2":
                    if hasattr(ss, "data2_prep"):
                        progressbar.progress(0.2)
                        wordcloud = vectorize_multiple([ss.data2_prep], extra_stopwords)
                        progressbar.progress(0.4)
                        wordcloud = visualize(wordcloud, cloud_color, cloud_bg, cloud_shape, cloud_font)
                        progressbar.progress(0.9)
                    else:
                        progressbar.progress(0.2)
                        ss.data2_prep = prep([ss.data2])
                        progressbar.progress(0.4)
                        wordcloud = vectorize_multiple([ss.data2_prep], extra_stopwords)
                        progressbar.progress(0.6)
                        wordcloud = visualize(wordcloud, cloud_color, cloud_bg, cloud_shape, cloud_font)
                        progressbar.progress(0.9) 
                if dataset == "Data 3":
                    if hasattr(ss, "data3_prep"):
                        progressbar.progress(0.2)
                        wordcloud = vectorize_multiple([ss.data3_prep], extra_stopwords)
                        progressbar.progress(0.4)
                        wordcloud = visualize(wordcloud, cloud_color, cloud_bg, cloud_shape, cloud_font)
                        progressbar.progress(0.9)
                    else:
                        progressbar.progress(0.2)
                        ss.data3_prep = prep([ss.data3])
                        progressbar.progress(0.4)
                        wordcloud = vectorize_multiple([ss.data3_prep], extra_stopwords)
                        progressbar.progress(0.6)
                        wordcloud = visualize(wordcloud, cloud_color, cloud_bg, cloud_shape, cloud_font)
                        progressbar.progress(0.9)
                if dataset == "Data 4":
                    if hasattr(ss, "data4_prep"):
                        progressbar.progress(0.2)
                        wordcloud = vectorize_multiple([ss.data4_prep], extra_stopwords)
                        progressbar.progress(0.4)
                        wordcloud = visualize(wordcloud, cloud_color, cloud_bg, cloud_shape, cloud_font)
                        progressbar.progress(0.9)
                    else:
                        progressbar.progress(0.2)
                        ss.data4_prep = prep([ss.data4])
                        progressbar.progress(0.4)
                        wordcloud = vectorize_multiple([ss.data4_prep], extra_stopwords)
                        progressbar.progress(0.6)
                        wordcloud = visualize(wordcloud, cloud_color, cloud_bg, cloud_shape, cloud_font)
                        progressbar.progress(0.9)
                if dataset == "Data 5":
                    if hasattr(ss, "data5_prep"):
                        progressbar.progress(0.2)
                        wordcloud = vectorize_multiple([ss.data5_prep], extra_stopwords)
                        progressbar.progress(0.4)
                        wordcloud = visualize(wordcloud, cloud_color, cloud_bg, cloud_shape, cloud_font)
                        progressbar.progress(0.9)
                    else:
                        progressbar.progress(0.2)
                        ss.data5_prep = prep([ss.data5])
                        progressbar.progress(0.4)
                        wordcloud = vectorize_multiple([ss.data5_prep], extra_stopwords)
                        progressbar.progress(0.6)
                        wordcloud = visualize(wordcloud, cloud_color, cloud_bg, cloud_shape, cloud_font)
                        progressbar.progress(0.9) 
                progress.header("Displaying wordcloud - this may take a second...")
                st.image(wordcloud.to_array())
                progressbar.progress(1.0)
                progress.header("Done! Save the wordcloud, or try changing the inputs for other results!")
        except AttributeError as e:
            st.write("Please scrape some data first, fool!")
            st.write("Error:", e)
    
    if type == "Compare Datasets":
        
        submit_button = False
        datanumber = 0

        for x in ['data1','data2','data3','data4','data5']:
            if hasattr(ss, x):
                datanumber += 1

        if datanumber == 0: 
            st.write("**You don't have any datasets loaded yet - comparison requires at least 2!**") 
            st.write("Please scrape or upload at least 2 datasets before continuing.")

        if datanumber == 1:
            st.write("**You only have 1 dataset loaded - comparison requires at least 2!**")
            st.write("Please scrape or upload at least 2 datasets before continuing.")
        
        if datanumber == 2:
            st.info("**2 datasets found in storage. Customize and create wordclouds below to compare!**")
            with st.form(key='2'):
                st.header("Customize wordclouds:")                  
                col1, col2 = st.beta_columns(2)

                cloud_color1 = col1.selectbox("Cloud 1 Theme:", ['Default (Black)','summer', 'Wistia', 'OrRd', 'YlGn'])
                cloud_font1 = col1.selectbox("Cloud 1 Font:", ['Default (AU Passata)','AU', 'SpicyRice'])
                cloud_bg1 = col1.selectbox("Cloud 1 Background:", ['Default (White)','black', 'white', 'red'])
                cloud_shape1 = col1.selectbox("Cloud 1 Shape:", ['Default (Square)','Circle', 'Heart'])

                cloud_color2 = col2.selectbox("Cloud 2 Theme:", ['Default (Black)','summer', 'Wistia', 'OrRd', 'YlGn'])
                cloud_font2 = col2.selectbox("Cloud 2 Font:", ['Default (AU Passata)','AU', 'SpicyRice'])
                cloud_bg2 = col2.selectbox("Cloud 2 Background:", ['Default (White)','black', 'white', 'red'])
                cloud_shape2 = col2.selectbox("Cloud 2 Shape:", ['Default (Square)','Circle', 'Heart'])
                
                extra_stopwords = st.text_input("Remove stopwords (please separate words by comma):")

                submit_button = st.form_submit_button(label='Create Wordclouds')

        if datanumber == 3:
            st.info("**3 datasets found in storage. Customize and create wordclouds below to compare!**")
            with st.form(key='3'):
                st.header("Customize wordclouds:")                  
                col1, col2, col3 = st.beta_columns(3)

                cloud_color1 = col1.selectbox("Cloud 1 Theme:", ['Default (Black)','summer', 'Wistia', 'OrRd', 'YlGn'])
                cloud_font1 = col1.selectbox("Cloud 1 Font:", ['Default (AU Passata)','AU', 'SpicyRice'])
                cloud_bg1 = col1.selectbox("Cloud 1 Background:", ['Default (White)','black', 'white', 'red'])
                cloud_shape1 = col1.selectbox("Cloud 1 Shape:", ['Default (Square)','Circle', 'Heart'])

                cloud_color2 = col2.selectbox("Cloud 2 Theme:", ['Default (Black)','summer', 'Wistia', 'OrRd', 'YlGn'])
                cloud_font2 = col2.selectbox("Cloud 2 Font:", ['Default (AU Passata)','AU', 'SpicyRice'])
                cloud_bg2 = col2.selectbox("Cloud 2 Background:", ['Default (White)','black', 'white', 'red'])
                cloud_shape2 = col2.selectbox("Cloud 2 Shape:", ['Default (Square)','Circle', 'Heart'])

                cloud_color3 = col3.selectbox("Cloud 3 Theme:", ['Default (Black)','summer', 'Wistia', 'OrRd', 'YlGn'])
                cloud_font3 = col3.selectbox("Cloud 3 Font:", ['Default (AU Passata)','AU', 'SpicyRice'])
                cloud_bg3 = col3.selectbox("Cloud 3 Background:", ['Default (White)','black', 'white', 'red'])
                cloud_shape3 = col3.selectbox("Cloud 3 Shape:", ['Default (Square)','Circle', 'Heart'])
                
                extra_stopwords = st.text_input("Remove stopwords (please separate words by comma):")
                
                submit_button = st.form_submit_button(label='Create Wordclouds')

        if datanumber == 4:
            st.write("**4 datasets found in storage. Customize and create wordclouds below to compare!**")
            with st.form(key='4'):
                st.header("Customize wordclouds:")                  
                col1, col2, col3, col4 = st.beta_columns(4)

                cloud_color1 = col1.selectbox("Cloud 1 Theme:", ['Default (Black)','summer', 'Wistia', 'OrRd', 'YlGn'])
                cloud_font1 = col1.selectbox("Cloud 1 Font:", ['Default (AU Passata)','AU', 'SpicyRice'])
                cloud_bg1 = col1.selectbox("Cloud 1 Background:", ['Default (White)','black', 'white', 'red'])
                cloud_shape1 = col1.selectbox("Cloud 1 Shape:", ['Default (Square)','Circle', 'Heart'])

                cloud_color2 = col2.selectbox("Cloud 2 Theme:", ['Default (Black)','summer', 'Wistia', 'OrRd', 'YlGn'])
                cloud_font2 = col2.selectbox("Cloud 2 Font:", ['Default (AU Passata)','AU', 'SpicyRice'])
                cloud_bg2 = col2.selectbox("Cloud 2 Background:", ['Default (White)','black', 'white', 'red'])
                cloud_shape2 = col2.selectbox("Cloud 2 Shape:", ['Default (Square)','Circle', 'Heart'])

                cloud_color3 = col3.selectbox("Cloud 3 Theme:", ['Default (Black)','summer', 'Wistia', 'OrRd', 'YlGn'])
                cloud_font3 = col3.selectbox("Cloud 3 Font:", ['Default (AU Passata)','AU', 'SpicyRice'])
                cloud_bg3 = col3.selectbox("Cloud 3 Background:", ['Default (White)','black', 'white', 'red'])
                cloud_shape3 = col3.selectbox("Cloud 3 Shape:", ['Default (Square)','Circle', 'Heart'])

                cloud_color4 = col4.selectbox("Cloud 4 Theme:", ['Default (Black)','summer', 'Wistia', 'OrRd', 'YlGn'])
                cloud_font4 = col4.selectbox("Cloud 4 Font:", ['Default (AU Passata)','AU', 'SpicyRice'])
                cloud_bg4 = col4.selectbox("Cloud 4 Background:", ['Default (White)','black', 'white', 'red'])
                cloud_shape4 = col4.selectbox("Cloud 4 Shape:", ['Default (Square)','Circle', 'Heart'])
                
                extra_stopwords = st.text_input("Remove stopwords (please separate words by comma):")
                
                submit_button = st.form_submit_button(label='Create Wordclouds')

        if datanumber == 5:
            st.write("**5 datasets found in storage. Customize and create wordclouds below to compare!**")
            with st.form(key='5'):
                st.header("Customize wordclouds:")                  
                col1, col2, col3, col4, col5 = st.beta_columns(5)

                cloud_color1 = col1.selectbox("Cloud 1 Theme:", ['Default (Black)','summer', 'Wistia', 'OrRd', 'YlGn'])
                cloud_font1 = col1.selectbox("Cloud 1 Font:", ['Default (AU Passata)','AU', 'SpicyRice'])
                cloud_bg1 = col1.selectbox("Cloud 1 Background:", ['Default (White)','black', 'white', 'red'])
                cloud_shape1 = col1.selectbox("Cloud 1 Shape:", ['Default (Square)','Circle', 'Heart'])

                cloud_color2 = col2.selectbox("Cloud 2 Theme:", ['Default (Black)','summer', 'Wistia', 'OrRd', 'YlGn'])
                cloud_font2 = col2.selectbox("Cloud 2 Font:", ['Default (AU Passata)','AU', 'SpicyRice'])
                cloud_bg2 = col2.selectbox("Cloud 2 Background:", ['Default (White)','black', 'white', 'red'])
                cloud_shape2 = col2.selectbox("Cloud 2 Shape:", ['Default (Square)','Circle', 'Heart'])

                cloud_color3 = col3.selectbox("Cloud 3 Theme:", ['Default (Black)','summer', 'Wistia', 'OrRd', 'YlGn'])
                cloud_font3 = col3.selectbox("Cloud 3 Font:", ['Default (AU Passata)','AU', 'SpicyRice'])
                cloud_bg3 = col3.selectbox("Cloud 3 Background:", ['Default (White)','black', 'white', 'red'])
                cloud_shape3 = col3.selectbox("Cloud 3 Shape:", ['Default (Square)','Circle', 'Heart'])

                cloud_color4 = col4.selectbox("Cloud 4 Theme:", ['Default (Black)','summer', 'Wistia', 'OrRd', 'YlGn'])
                cloud_font4 = col4.selectbox("Cloud 4 Font:", ['Default (AU Passata)','AU', 'SpicyRice'])
                cloud_bg4 = col4.selectbox("Cloud 4 Background:", ['Default (White)','black', 'white', 'red'])
                cloud_shape4 = col4.selectbox("Cloud 4 Shape:", ['Default (Square)','Circle', 'Heart'])

                cloud_color5 = col5.selectbox("Cloud 5 Theme:", ['Default (Black)','summer', 'Wistia', 'OrRd', 'YlGn'])
                cloud_font5 = col5.selectbox("Cloud 5 Font:", ['Default (AU Passata)','AU', 'SpicyRice'])
                cloud_bg5 = col5.selectbox("Cloud 5 Background:", ['Default (White)','black', 'white', 'red'])
                cloud_shape5 = col5.selectbox("Cloud 5 Shape:", ['Default (Square)','Circle', 'Heart'])
                
                extra_stopwords = st.text_input("Remove stopwords (please separate words by comma):")
                
                submit_button = st.form_submit_button(label='Create Wordclouds')

        progress = st.header("Ready to plot! Click 'Create Wordcloud' to begin.")
        progressbar = st.progress(0)

        try:
            if submit_button:
                progress.header("Building wordclouds, please wait...")
                progressbar.progress(0.1)

                if datanumber == 2:
                    df_list = [ss.data1,ss.data2]
                    cloud_color_list = [cloud_color1, cloud_color2]
                    cloud_bg_list = [cloud_bg1, cloud_bg2]
                    cloud_shape_list = [cloud_shape1, cloud_shape2]
                    cloud_font_list = [cloud_font1, cloud_font2]

                if datanumber == 3:
                    df_list = [ss.data1,ss.data2, ss.data3]
                    cloud_color_list = [cloud_color1, cloud_color2, cloud_color3]
                    cloud_bg_list = [cloud_bg1, cloud_bg2, cloud_bg3]
                    cloud_shape_list = [cloud_shape1, cloud_shape2, cloud_shape3]
                    cloud_font_list = [cloud_font1, cloud_font2, cloud_font3]

                if datanumber == 4:
                    df_list = [ss.data1,ss.data2, ss.data3, ss.data4]
                    cloud_color_list = [cloud_color1, cloud_color2, cloud_color3, cloud_color4]
                    cloud_bg_list = [cloud_bg1, cloud_bg2, cloud_bg3, cloud_bg4]
                    cloud_shape_list = [cloud_shape1, cloud_shape2, cloud_shape3, cloud_shape4]
                    cloud_font_list = [cloud_font1, cloud_font2, cloud_font3, cloud_font4]

                if datanumber == 5:
                    df_list = [ss.data1,ss.data2, ss.data3, ss.data4, ss.data5]
                    cloud_color_list = [cloud_color1, cloud_color2, cloud_color3, cloud_color4, cloud_color5]
                    cloud_bg_list = [cloud_bg1, cloud_bg2, cloud_bg3, cloud_bg4, cloud_bg5]
                    cloud_shape_list = [cloud_shape1, cloud_shape2, cloud_shape3, cloud_shape4, cloud_shape5]
                    cloud_font_list = [cloud_font1, cloud_font2, cloud_font3, cloud_font4, cloud_font5]

                progressbar.progress(0.2)
                prepped_list = []
                progress_value = 0.2
                progress.header("Preprocessing...")

                for i in df_list:
                    increment = 0.2/len(df_list)
                    progress_value += increment 
                    progressbar.progress(progress_value)
                    if hasattr(ss, "data1"):
                        if str(i) == str(ss.data1):
                            if hasattr(ss, "data1_prep"):
                                prepped_list.append(ss.data1_prep)
                                continue
                            else:
                                ss.data1_prep = prep([i])
                                prepped_list.append(ss.data1_prep)

                    if hasattr(ss, "data2"):
                        if str(i) == str(ss.data2):
                            if hasattr(ss, "data2_prep"):
                                prepped_list.append(ss.data2_prep)
                                continue
                            else:
                                ss.data2_prep = prep([i])
                                prepped_list.append(ss.data2_prep)
                    
                    if hasattr(ss, "data3"):
                        if str(i) == str(ss.data3):
                            if hasattr(ss, "data3_prep"):
                                prepped_list.append(ss.data3_prep)
                                continue
                            else:
                                ss.data3_prep = prep([i])
                                prepped_list.append(ss.data3_prep)
                    
                    if hasattr(ss, "data4"):
                        if str(i) == str(ss.data4):
                            if hasattr(ss, "data4_prep"):
                                prepped_list.append(ss.data4_prep)
                                continue
                            else:
                                ss.data4_prep = prep([i])
                                prepped_list.append(ss.data4_prep)
                    
                    if hasattr(ss, "data5"):
                        if str(i) == str(ss.data5):
                            if hasattr(ss, "data5_prep"):
                                prepped_list.append(ss.data5_prep)
                                continue
                            else:
                                ss.data5_prep = prep([i])
                                prepped_list.append(ss.data5_prep)


                prepped_list = [item for sublist in prepped_list for item in sublist]
                progress.header("Vectorizing...")
                df = vectorize_multiple(prepped_list, extra_stopwords)
                progressbar.progress(0.5)
                total_dfs = len(prepped_list)
                wordcloud_list = []

                progress.header("Visualizing...")
                progress_value = 0.5
                for x in range(total_dfs):
                    increment = 0.4/len(df_list)
                    progress_value += increment 
                    progressbar.progress(progress_value)
                    selected_df = df[[x]]
                    cloud_color = cloud_color_list[x]
                    cloud_bg = cloud_bg_list[x]
                    cloud_shape = cloud_shape_list[x]
                    cloud_font = cloud_font_list[x]
                    wordcloud = visualize(selected_df, cloud_color, cloud_bg, cloud_shape, cloud_font, x)
                    wordcloud_list.append(wordcloud)
                
                windows = len(df_list)
                
                progressbar.progress(0.95)
                progress.header("Displaying wordclouds - this may take a second...")

                if windows == 2:
                    wc_col1, wc_col2 = st.beta_columns(windows)
                    wc_col1.text(ss.data1q)
                    wc_col2.text(ss.data2q)
                    wc_col1.image(wordcloud_list[0].to_array())
                    wc_col2.image(wordcloud_list[1].to_array())
                if windows == 3:
                    wc_col1, wc_col2, wc_col3 = st.beta_columns(windows)
                    wc_col1.text(ss.data1q)
                    wc_col2.text(ss.data2q)
                    wc_col3.text(ss.data3q)
                    wc_col1.image(wordcloud_list[0].to_array())
                    wc_col2.image(wordcloud_list[1].to_array())
                    wc_col3.image(wordcloud_list[2].to_array())
                if windows == 4:
                    wc_col1, wc_col2, wc_col3, wc_col4 = st.beta_columns(windows)
                    wc_col1.text(ss.data1q)
                    wc_col2.text(ss.data2q)
                    wc_col3.text(ss.data3q)
                    wc_col4.text(ss.data4q)
                    wc_col1.image(wordcloud_list[0].to_array())
                    wc_col2.image(wordcloud_list[1].to_array())
                    wc_col3.image(wordcloud_list[2].to_array())
                    wc_col4.image(wordcloud_list[3].to_array())
                if windows == 5:
                    wc_col1, wc_col2, wc_col3, wc_col4, wc_col5 = st.beta_columns(windows)
                    wc_col1.text(ss.data1q)
                    wc_col2.text(ss.data2q)
                    wc_col3.text(ss.data3q)
                    wc_col4.text(ss.data4q)
                    wc_col5.text(ss.data5q)
                    wc_col1.image(wordcloud_list[0].to_array())
                    wc_col2.image(wordcloud_list[1].to_array())
                    wc_col3.image(wordcloud_list[2].to_array())
                    wc_col4.image(wordcloud_list[3].to_array())
                    wc_col5.image(wordcloud_list[4].to_array())
                time.sleep(0.5)
                progressbar.progress(1.0)
                progress.header("Done! Save the wordclouds, or try changing the inputs for other results!")
        except AttributeError as e:
            st.write("Please scrape some data first, fool!")
            st.write("Error:", e)

def page_sentiment():
    st.title("IT'S ABOUT EMOTIONS, FUCKER")
    st.text("Let's see how we feel.")

    #Generate sentiment scores
    #generate list of dataframes:
    list_dfs = []
    queries = []

    if hasattr(ss, 'data1'):
        ss.data1['scores'] = ss.data1['Comment'].apply(lambda comment: sid.polarity_scores(comment))
        ss.data1['compound'] = ss.data1['scores'].apply(lambda x: x.get('compound'))
        list_dfs.append(ss.data1)
        queries.append(ss.data1q)
    if hasattr(ss, 'data2'):
        ss.data2['scores'] = ss.data2['Comment'].apply(lambda comment: sid.polarity_scores(comment))
        ss.data2['compound'] = ss.data2['scores'].apply(lambda x: x.get('compound'))
        list_dfs.append(ss.data2)
        queries.append(ss.data2q)
    if hasattr(ss, 'data3'):
        ss.data3['scores'] = ss.data3['Comment'].apply(lambda comment: sid.polarity_scores(comment))
        ss.data3['compound'] = ss.data3['scores'].apply(lambda x: x.get('compound'))
        list_dfs.append(ss.data3)
        queries.append(ss.data3q)
    if hasattr(ss, 'data4'):
        ss.data4['scores'] = ss.data4['Comment'].apply(lambda comment: sid.polarity_scores(comment))
        ss.data4['compound'] = ss.data4['scores'].apply(lambda x: x.get('compound'))
        list_dfs.append(ss.data4)
        queries.append(ss.data4q)
    if hasattr(ss, 'data5'):
        ss.data5['scores'] = ss.data5['Comment'].apply(lambda comment: sid.polarity_scores(comment))
        ss.data5['compound'] = ss.data5['scores'].apply(lambda x: x.get('compound'))
        list_dfs.append(ss.data5)
        queries.append(ss.data5q)
    

    if len(queries) == 0:
            st.write("**You don't have any datasets loaded yet - analysis requires at least 1!**") 
            st.write("Please scrape or upload at least 1 dataset before continuing.")
    else:
            
        formated = pd.concat(list_dfs)

        formated["Comment Published"] = pd.to_datetime(formated["Comment Published"], format= "%Y-%m-%dT%H:%M:%SZ")
        formated["Comment Published"] = pd.to_datetime(formated["Comment Published"], format='%Y-%m-%d')
        formated["Comment Published"] = formated["Comment Published"].dt.strftime('%Y-%m-%d')

        formated = formated.groupby(['Query', 'Comment Published'], as_index=False)['compound'].mean()
        st.dataframe(formated)
        options = st.multiselect(
        'Select data to include',
        queries,
        default=queries[0])

        formated = formated[formated.Query.isin(options)]
        
        brush = alt.selection(type='interval', encodings=['x'])
        base = alt.Chart(formated).mark_line().encode(
        x = 'Comment Published:T',
        y = 'compound:Q',
        color=alt.Color("Query")
        ).properties(
        width=1000,
        height=400
        )

        upper = base.encode(
        alt.X('Comment Published:T', scale=alt.Scale(domain=brush))
        )

        lower = base.properties(
        height=60
        ).add_selection(brush)

        chart2 = upper & lower

        st.altair_chart(chart2, use_container_width=True)


def page_topic():
    st.title("TOPICS ARE COOL. LET'S ANALYZE THEM.")


def page_about():
    st.title("LET'S TALK A LITTLE ABOUT US")
    st.text("We made this, so we're pretty cool, hey.")   


if __name__ == "__main__":
    main()



