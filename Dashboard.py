import SessionState
from scraper_fns import *
from nlp_fns import *
import base64
import altair as alt
import time
from sklearn.feature_extraction.text import TfidfVectorizer
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from scipy import sparse
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk
nltk.download('stopwords')
import numpy as np
from nltk.corpus import stopwords
import gensim
from gensim.utils import simple_preprocess
import spacy
import re 
import os 
import pickle 
from pandas import DataFrame
#import google.oauth2.credentials
# from googleapiclient.discovery import build
# from googleapiclient.errors import HttpError
# from google_auth_oauthlib.flow import InstalledAppFlow
# from google.auth.transport.requests import Request
from youtube_transcript_api import YouTubeTranscriptApi
from sklearn.feature_extraction.text import TfidfVectorizer
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from PIL import Image
from enum import Enum
from io import BytesIO, StringIO
from typing import Union
import pandas as pd
import streamlit as st


ss = SessionState.get(current = "Welcome", upload = "NA", wc = "NA", 
                df_list = [1,1,1,1,1], prep_list = [1,1,1,1,1],
                query_list = [1,1,1,1,1], length_list = [0,0,0,0,0],
                href_list = [1,1,1,1,1])

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
        "Manage Stored Data": page_manage,
        "Wordcloud Analysis" : page_visualize,
        "Sentiment Analysis" : page_sentiment,
        "Topic Analysis" : page_topic
   #     "About Us" : page_about
    }
    # Sidebar title
    st.sidebar.image("./resources/logo.png", use_column_width=True)
    st.sidebar.title("Navigation")
    st.sidebar.header("")

    # Sidebar buttons - if a button is True, state will be set to the page of that button

    with st.sidebar.beta_expander("About This App", expanded = True):
        if st.button("Welcome To YouNLP"):
            ss.current = "Welcome"
        if st.button("How Does It Work?"):
            ss.current = "How it works"
       # if st.button("About Us"):
       #     ss.current = "About Us"

    with st.sidebar.beta_expander("Manage Data", expanded = True):
        if st.button("Scrape YouTube Data"):
            ss.current = "Scrape YouTube Data"
        if st.button("Upload YouTube Data"):
            ss.current = "Upload Data"
        if st.button("Manage Stored Data"):
            ss.current = "Manage Stored Data"

    with st.sidebar.beta_expander("Analyze Data", expanded = True):
        if st.button("Wordcloud Analysis"):
            ss.current = "Wordcloud Analysis"
        if st.button("Sentiment Analysis"):
            ss.current = "Sentiment Analysis"
        if st.button("Similarity Analysis"):
            ss.current = "Topic Analysis"

    with st.sidebar.beta_expander("Data Storage", expanded = True):
        for index in [0,1,2,3,4]:
            if str(ss.df_list[index]) == "1":
                st.markdown('<font color=grey>**CONTAINER ' + str(index+1) + ':** \n *Not in use*</font>', unsafe_allow_html=True)
            else:
                st.markdown('<font color=green>**CONTAINER ' + str(index+1) + ':**</font>', unsafe_allow_html=True)
                datatext = "**Search term: **" + str(ss.query_list[index]) + "  \n   **Comments:** " + str(ss.length_list[index])
                st.write(datatext)
                st.markdown(ss.href_list[index], unsafe_allow_html=True)

    # Display the selected page with the session state
    pages[ss.current]()

def page_main():
    st.title('Welcome to YouNLP')
    st.header("**Catchy header**")
    st.write("This is where the welcome text fucking goes, motherfucker.")

def page_how():
    st.header('**How To Use This Tool**')
    col1, col2 = st.beta_columns([1,1])
    col1.write("This is where we put a quick, short summary of the tool.")
    col2.image("./resources/screenshots/flowchart.png")

    st.subheader("**Manage Data**")
    st.write("Everything that has to do with data management is located in the *Manage Data* menu of the sidebar. Here, you can scrape new data from YouTube, upload previously scraped data, and view or delete data currently stored in the app.")
    st.info("Click one of the drop-down menus below to learn more about the tools in the Manage Data menu!")
    
    with st.beta_expander("Scrape YouTube Data", expanded = False):
        st.subheader("*Scrape YouTube Data*")
        col1,col2 = st.beta_columns([1,1])
        col1.write("The *Scrape YouTube Data* tool allows you to create a new dataset for analysis based on any search term. In the tool's menu, you have three")
        col2.image("./resources/screenshots/scrape.png")

    with st.beta_expander("Upload YouTube Data", expanded = False):
        st.subheader("*Upload YouTube Data*")
        col1,col2 = st.beta_columns([1,1])
        col1.write("The *Scrape YouTube Data* tool allows you to create a new dataset for analysis based on any search term. In the tool's menu, you have three")
        col2.image("./resources/screenshots/upload.png")

    with st.beta_expander("Manage Stored Data", expanded = False):
        st.subheader("*Manage Stored Data*")
        col1,col2 = st.beta_columns([1,1])
        col1.write("The *Scrape YouTube Data* tool allows you to create a new dataset for analysis based on any search term. In the tool's menu, you have three")
        col2.image("./resources/screenshots/manage.png")

    st.subheader("**Analyze Data**")
    st.write("Everything that has to do with data analysis and visualization is located in the *Analyze Data* menu of the sidebar. Here, you can do a variety of different types of language analysis.")
    st.info("Click one of the drop-down menus below to learn more about the tools in the Analyze Data menu!")
    
    with st.beta_expander("Wordcloud Analysis", expanded = False):
        st.subheader("*Wordcloud Analysis*")
        st.write("The Word")
        col1,col2 = st.beta_columns([1,1])
        col1.write("The *Scrape YouTube Data* tool allows you to create a new dataset for analysis based on any search term. In the tool's menu, you have three")
        col2.image("./resources/screenshots/wc_single.png")

        col1.write("The *Scrape YouTube Data* tool allows you to create a new dataset for analysis based on any search term. In the tool's menu, you have three")
        col2.image("./resources/screenshots/wc_multiple.png")

    with st.beta_expander("Sentiment Analysis", expanded = False):
        st.subheader("*Sentiment Analysis*")
        col1,col2 = st.beta_columns([1,1])
        col1.write("The *Scrape YouTube Data* tool allows you to create a new dataset for analysis based on any search term. In the tool's menu, you have three")
        col2.image("./resources/screenshots/upload.png")

    with st.beta_expander("Similarity Analysis", expanded = False):
        st.subheader("*Similarity Analysis*")
        col1,col2 = st.beta_columns([1,1])
        col1.write("The *Scrape YouTube Data* tool allows you to create a new dataset for analysis based on any search term. In the tool's menu, you have three")
        col2.image("./resources/screenshots/manage.png")



def page_upload():

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


   # """Run this function to display the Streamlit app"""
    st.header('**Upload YouTube Data**')
    st.markdown(STYLE, unsafe_allow_html=True)

    with st.form(key='my_form'):
        col1, col2, col3 = st.beta_columns([1,2,3])
        col3.write("**Upload existing data for analysis!**  \n If you have an existing dataset from a previous YouTube search, you can use this tool to upload it for further analysis or comparison with other datasets. The uploaded dataset is stored in the selected container - make sure to check that you're not overwriting other necessary data!")
        containers = ['Container 1', 'Container 2', 'Container 3', 'Container 4', 'Container 5']
        dataset = col1.radio("Upload data to:", (containers))
        df_index = int(dataset[-1])-1
        file = col2.file_uploader("Upload CSV file", type="CSV")
      #  show_file = col2.empty()
        st.info("To upload existing data, please select a CSV file from the browser and a container to store the data before clicking Upload File. Note that uploaded data must originate from this scraping tool, and/or have the same format.")
        submit_button = st.form_submit_button(label='Upload File')
        if not file:
        #    show_file.info("Upload CSV")
            return
        file_type = get_file_type(file)

    if submit_button:
            
        if file_type == FileType.CSV:

            ss.df_list[df_index] = pd.read_csv(file)
            ss.query_list[df_index] = ss.df_list[df_index].at[2,'Query']
            ss.length_list[df_index] = len(ss.df_list[df_index])
            csv = ss.df_list[df_index].to_csv(index=False)
            b64 = base64.b64encode(csv.encode()).decode()  # some strings <-> bytes conversions necessary here
            ss.href_list[df_index] = f'<a href="data:file/csv;base64,{b64}" download="{ss.query_list[df_index]}.csv">Download CSV</a>'
            st.info(f"File using the query '{ss.query_list[df_index]}' with {ss.length_list[df_index]} comments successfully uploaded to Data Container {df_index+1}")
            with st.beta_expander("Examine data frame"):
                st.dataframe(ss.df_list[df_index].head(10))
            ss.prep_list[df_index] = 1

    file.close()

    
def page_scrape():
    st.header('**Scrape Data From YouTube**')
    with st.form(key='my_form'):
        col1, col2, col3 = st.beta_columns([1,2,3])
        col3.write("**To analyse data, you gotta have data first!**  \n This tool executes a YouTube search for an input query, finds the number of videos selected, and scrapes up to the top 100 comments of each video. Details about the scraped videos and comments are then converted to a data frame and stored in the selected container in the app. After scraping is complete, feel free to download the data frame or use one of the tools in the sidebar for further analysis. You can store up to 5 datasets in the app's containers.")
        containers = ['Container 1', 'Container 2', 'Container 3', 'Container 4', 'Container 5']
        dataset = col1.radio("Stored scraped data in:", (containers))
        df_index = int(dataset[-1])-1
        user_input1 = col2.text_input("Search term:", '')
        user_input2 = col2.number_input("Videos:", 1)
        st.info("Please note: YouTube's API has a daily limit of 10,000 requests. Please limit searches to 100 videos or less.")
        submit_button = st.form_submit_button(label='Start Scraper')

    if submit_button:
        if user_input1 == "":
            st.info("Please enter a search term before scraping!")
        else:
            progress = st.subheader("Scraping YouTube, please wait...")

            ss.df_list[df_index] = get_data(user_input1, user_input2)
            ss.query_list[df_index] = user_input1
            ss.length_list[df_index] = len(ss.df_list[df_index])
            csv = ss.df_list[df_index].to_csv(index=False)
            b64 = base64.b64encode(csv.encode()).decode()  # some strings <-> bytes conversions necessary here
            ss.href_list[df_index] = f'<a href="data:file/csv;base64,{b64}" download="{ss.query_list[df_index]}.csv">Download CSV</a>'
            st.markdown(ss.href_list[df_index], unsafe_allow_html=True)
            with st.beta_expander("Examine data frame"):
                st.dataframe(ss.df_list[df_index].head(10))
            ss.prep_list[df_index] = 1

            progress.subheader("Done! Total comments scraped: " + str(ss.length_list[df_index]))


def page_manage():
    st.header('**Manage Stored Data**')

    st.subheader("Container Status")
    st.info("Here, you can examine, download, delete, or load in sample data to the app's containers. Please note that once a container is emptied or overwritten, the data previously stored in that container is permanently deleted. Using the links under each container, you can download datasets and later reupload them for additional analysis or comparison.")
    col1, col2, col3, col4, col5 = st.beta_columns(5)
    for index, col in zip([0,1,2,3,4], [col1, col2, col3, col4, col5]):
        if str(ss.df_list[index]) == "1":
            col.markdown('<font color=grey>**CONTAINER ' + str(index+1) + ':** \n *Not in use*</font>', unsafe_allow_html=True)
        else:
            col.markdown('<font color=green>**CONTAINER ' + str(index+1) + ':**</font>', unsafe_allow_html=True)
            datatext = "**Search term: **" + str(ss.query_list[index]) + "  \n   **Comments:** " + str(ss.length_list[index])
            col.write(datatext)
            col.markdown(ss.href_list[index], unsafe_allow_html=True)

    with st.form(key='my_form'):
        containers = ['Container 1', 'Container 2', 'Container 3', 'Container 4', 'Container 5']
        dataset = st.radio("Select container to empty:", (containers)) 
        df_index = int(dataset[-1])-1
        
        submit_button = st.form_submit_button(label='Empty selected container')

        
    if submit_button:
        if str(ss.df_list[df_index]) == str(1):
            message = "Container is already empty."
        else:
            message = "Removed data in Container " + str(df_index+1)
        ss.df_list[df_index] = 1
        ss.prep_list[df_index] = 1
        ss.query_list[df_index] = 1
        ss.href_list[df_index] = 1
        ss.length_list[df_index] = 0

        st.info(message)

    st.subheader("Examine Stored Datasets")

    for index in [0,1,2,3,4]:
        if str(ss.df_list[index]) != "1":
            with st.beta_expander("Examine data in Container " + str(index+1), expanded = False):
                st.dataframe(ss.df_list[index].head(10))
    
  #  st.subheader(" Stored Datasets")


    with st.form(key='my_form2'):
        col1, col2, col3 = st.beta_columns([1,1,2])
        st.subheader("Load Sample Data")
        containers = ['Container 1', 'Container 2', 'Container 3', 'Container 4', 'Container 5']
        testsets = ['Dogecoin', 'Bitcoin', 'Biden', 'Trump', 'NLP']
        testdata = col2.radio("Choose a sample dataset:", (testsets)) 
        dataset = col1.radio("Select container to upload sample data:", (containers)) 
        df_index = int(dataset[-1])-1

        col3.write("**Test out the app without scraping!**")
        col3.write("If you want to take the various tools out for a spin without going through the scraping or uploading process, try loading in one of our existing data samples for exploration. These datasets have been meticulously curated for the linguistic connoisseur, giving exciting insights into interesting topics such as memes, cryptocurrencies, and language analytics!")
    #   for index, col in zip([0,1,2,3,4], [col1, col2, col3, col4, col5]):
    #       if str(ss.df_list[index]) == "1":
     #           col.markdown('<font color=grey>**CONTAINER ' + str(index+1) + ':** \n *Not in use*</font>', unsafe_allow_html=True)
    #       else:
    #           col.markdown('<font color=green>**CONTAINER ' + str(index+1) + ':**</font>', unsafe_allow_html=True)
     #           datatext = "**Search term: **" + str(ss.query_list[index]) + "  \n   **Comments:** " + str(ss.length_list[index])
    #           col.write(datatext)
    #           col.markdown(ss.href_list[index], unsafe_allow_html=True)
        submit_button2 = st.form_submit_button(label='Load Test Data')

        
    if submit_button2:
        st.write("Testdata = " ,testdata)
    #   if str(ss.df_list[df_index]) != str(1):
    #       st.info("Container is already in use! To avoid accidental overwriting, please choose another container or empty the selected container.")
    #   else:
  #      ss.df_list[df_index] = pd.read_csv("./resources/testdata/" + testdata + ".csv")
  #      ss.prep_list[df_index] = 1
  #      ss.query_list[df_index] = ss.df_list[df_index].at[2,'Query']
  #      ss.href_list[df_index] = f'<a href="data:file/csv;base64,{b64}" download="{ss.query_list[df_index]}.csv">Download CSV</a>'
  #      ss.length_list[df_index] = len(ss.df_list[df_index])
  #      st.info(f"File using the query '{ss.query_list[df_index]}' with {ss.length_list[df_index]} comments successfully uploaded to Data Container {df_index+1}")


def page_visualize():

    cloud_color_choices = ['Summer','Autumn', 'Winter', 'Spring', 'Gray']
    cloud_bg_choices = ['Transparent','Black', 'White', 'Gray', 'Beige']
    cloud_font_choices = ['AU Passata','Spicy Rice', 'Raleway', 'Kenyan Coffee', 'Comic Sans']
    cloud_shape_choices = ['Square','Circle', 'Heart', 'Brain', 'Mushroom']

    st.header('**Wordcloud Visualization and Comparison**')
    col1, col2 = st.beta_columns(2)
    type = col1.selectbox(label = "",options = ['Individual Datasets (Word Frequency)', 'Compare Datasets (TF-IDF Scores)'])
    st.info("Use the menu above to switch between wordclouds examining individual datasets (using word frequency) and wordclouds comparing multiple datasets (using TF-IDF scores). For more information on both, see the descriptions for each type in the information box on the right.")   

    if type == "Individual Datasets (Word Frequency)":
        datanumber = 0
        for index in [0,1,2,3,4]:
            if str(ss.df_list[index]) != str(1):
                datanumber += 1

        if datanumber == 0: 
            st.write("**You don't have any datasets loaded yet - wordclouds require at least 1!**") 
            st.write("Please scrape or upload at least 1 dataset before continuing.")

        if datanumber > 0:

            with st.form(key='my_form'):
                col1, col2, col3, col4 = st.beta_columns([2,3,3,4])
                containers = ['Container 1', 'Container 2', 'Container 3', 'Container 4', 'Container 5']
                dataset = col1.radio("Choose data to plot:", (containers))
                df_index = int(dataset[-1])-1
                dataset = ss.df_list[df_index] 
                cloud_color = col2.selectbox("Theme:", cloud_color_choices)
                cloud_bg = col2.selectbox("Background:", cloud_bg_choices)
                cloud_font = col3.selectbox("Font:", cloud_font_choices)
                cloud_shape = col3.selectbox("Shape:", cloud_shape_choices)

                col4.write("**They're clouds, but made from words!**  \n Wordclouds are one of the most simple yet effective visualizations of large amounts of text data. A tag cloud (word cloud or wordle or weighted list in visual design) is a novelty visual representation of text data, typically used to depict keyword metadata (tags) on websites, or to visualize free form text.")   
                col1, col2 = st.beta_columns([4,2])
                extra_stopwords = col1.text_input("Remove stopwords (please separate words by comma):")
                col2.write("**Wordclouds from word frequencies**  \n Wordclouds in this tool are made from word frequencies, where the size of a word in the cloud corresponds to the number of times the word is mentioned in the analyzed data.")
                submit_button = col1.form_submit_button(label='Create Wordcloud')
            progress = st.header("Ready to plot! Click 'Create Wordcloud' to begin.")
            progressbar = st.progress(0)

            try:
                if submit_button:
                    progress.header("Building wordcloud, please wait...")
                    progressbar.progress(0.1)
                    
                    if ss.prep_list[df_index] != 1:
                        progressbar.progress(0.2)
                        wordcloud = vectorize_multiple([ss.prep_list[df_index]], extra_stopwords)
                        progressbar.progress(0.3)
                        wordcloud = visualize(wordcloud, cloud_color, cloud_bg, cloud_shape, cloud_font)
                        progressbar.progress(0.9)
                    else:
                        progressbar.progress(0.2)
                        ss.prep_list[df_index] = prep([ss.df_list[df_index]])
                        progressbar.progress(0.3)
                        wordcloud = vectorize_multiple([ss.prep_list[df_index]], extra_stopwords)
                        progressbar.progress(0.5)
                        wordcloud = visualize(wordcloud, cloud_color, cloud_bg, cloud_shape, cloud_font)
                        progressbar.progress(0.9)

                    progress.header("Displaying wordcloud - this may take a second...")
                    subcol1,subcol2,subcol3 = st.beta_columns([2,4,2])
                    subcol2.subheader("Search term: *" + ss.query_list[df_index] + "*")
                    subcol2.image(wordcloud.to_array())
                    progressbar.progress(1.0)
                    progress.header("Done! Save the wordcloud, or try changing the inputs for other results!")
            except AttributeError as e:
                st.write("Please scrape some data first, fool!")
                st.write("Error:", e)


    if type == "Compare Datasets (TF-IDF Scores)":
        
        submit_button = False
        datanumber = 0

        for index in [0,1,2,3,4]:
            if str(ss.df_list[index]) != str(1):
                datanumber += 1

        if datanumber == 0: 
            st.write("**You don't have any datasets loaded yet - wordcloud comparison requires at least 2!**") 
            st.write("Please scrape or upload at least 2 datasets before continuing.")

        if datanumber == 1:
            st.write("**You only have 1 dataset loaded - wordcloud comparison requires at least 2!**")
            st.write("Please scrape or upload at least 2 datasets before continuing.")
        
        if datanumber == 2:
            st.info("**2 datasets found in storage. Customize and create wordclouds below to compare!**")
            with st.form(key='2'):
                st.header("Customize wordclouds:")                  
                col1, col2 = st.beta_columns(2)

                cloud_color1 = col1.selectbox("Cloud 1 Theme:", cloud_color_choices)
                cloud_font1 = col1.selectbox("Cloud 1 Font:", cloud_font_choices)
                cloud_bg1 = col1.selectbox("Cloud 1 Background:", cloud_bg_choices)
                cloud_shape1 = col1.selectbox("Cloud 1 Shape:", cloud_shape_choices)

                cloud_color2 = col2.selectbox("Cloud 2 Theme:", cloud_color_choices)
                cloud_font2 = col2.selectbox("Cloud 2 Font:", cloud_font_choices)
                cloud_bg2 = col2.selectbox("Cloud 2 Background:", cloud_bg_choices)
                cloud_shape2 = col2.selectbox("Cloud 2 Shape:", cloud_shape_choices)
                
                extra_stopwords = st.text_input("Remove stopwords (please separate words by comma):")

                submit_button = st.form_submit_button(label='Create Wordclouds')

            progress = st.header("Ready to plot! Click 'Create Wordcloud' to begin.")
            progressbar = st.progress(0)

        if datanumber == 3:
            st.info("**3 datasets found in storage. Customize and create wordclouds below to compare!**")
            with st.form(key='3'):
                st.header("Customize wordclouds:")                  
                col1, col2, col3 = st.beta_columns(3)

                cloud_color1 = col1.selectbox("Cloud 1 Theme:", cloud_color_choices)
                cloud_font1 = col1.selectbox("Cloud 1 Font:", cloud_font_choices)
                cloud_bg1 = col1.selectbox("Cloud 1 Background:", cloud_bg_choices)
                cloud_shape1 = col1.selectbox("Cloud 1 Shape:", cloud_shape_choices)

                cloud_color2 = col2.selectbox("Cloud 2 Theme:", cloud_color_choices)
                cloud_font2 = col2.selectbox("Cloud 2 Font:", cloud_font_choices)
                cloud_bg2 = col2.selectbox("Cloud 2 Background:", cloud_bg_choices)
                cloud_shape2 = col2.selectbox("Cloud 2 Shape:", cloud_shape_choices)

                cloud_color3 = col3.selectbox("Cloud 3 Theme:", cloud_color_choices)
                cloud_font3 = col3.selectbox("Cloud 3 Font:", cloud_font_choices)
                cloud_bg3 = col3.selectbox("Cloud 3 Background:", cloud_bg_choices)
                cloud_shape3 = col3.selectbox("Cloud 3 Shape:", cloud_shape_choices)
                
                extra_stopwords = st.text_input("Remove stopwords (please separate words by comma):")
                
                submit_button = st.form_submit_button(label='Create Wordclouds')

            progress = st.header("Ready to plot! Click 'Create Wordcloud' to begin.")
            progressbar = st.progress(0)

        if datanumber == 4:
            st.write("**4 datasets found in storage. Customize and create wordclouds below to compare!**")
            with st.form(key='4'):
                st.header("Customize wordclouds:")                  
                col1, col2, col3, col4 = st.beta_columns(4)

                cloud_color1 = col1.selectbox("Cloud 1 Theme:", cloud_color_choices)
                cloud_font1 = col1.selectbox("Cloud 1 Font:", cloud_font_choices)
                cloud_bg1 = col1.selectbox("Cloud 1 Background:", cloud_bg_choices)
                cloud_shape1 = col1.selectbox("Cloud 1 Shape:", cloud_shape_choices)

                cloud_color2 = col2.selectbox("Cloud 2 Theme:", cloud_color_choices)
                cloud_font2 = col2.selectbox("Cloud 2 Font:", cloud_font_choices)
                cloud_bg2 = col2.selectbox("Cloud 2 Background:", cloud_bg_choices)
                cloud_shape2 = col2.selectbox("Cloud 2 Shape:", cloud_shape_choices)

                cloud_color3 = col3.selectbox("Cloud 3 Theme:", cloud_color_choices)
                cloud_font3 = col3.selectbox("Cloud 3 Font:", cloud_font_choices)
                cloud_bg3 = col3.selectbox("Cloud 3 Background:", cloud_bg_choices)
                cloud_shape3 = col3.selectbox("Cloud 3 Shape:", cloud_shape_choices)

                cloud_color4 = col4.selectbox("Cloud 4 Theme:", cloud_color_choices)
                cloud_font4 = col4.selectbox("Cloud 4 Font:", cloud_font_choices)
                cloud_bg4 = col4.selectbox("Cloud 4 Background:", cloud_bg_choices)
                cloud_shape4 = col4.selectbox("Cloud 4 Shape:", cloud_shape_choices)
                
                extra_stopwords = st.text_input("Remove stopwords (please separate words by comma):")
                
                submit_button = st.form_submit_button(label='Create Wordclouds')

            progress = st.header("Ready to plot! Click 'Create Wordcloud' to begin.")
            progressbar = st.progress(0)

        if datanumber == 5:
            st.write("**5 datasets found in storage. Customize and create wordclouds below to compare!**")
            with st.form(key='5'):
                st.header("Customize wordclouds:")                  
                col1, col2, col3, col4, col5 = st.beta_columns(5)

                cloud_color1 = col1.selectbox("Cloud 1 Theme:", cloud_color_choices)
                cloud_font1 = col1.selectbox("Cloud 1 Font:", cloud_font_choices)
                cloud_bg1 = col1.selectbox("Cloud 1 Background:", cloud_bg_choices)
                cloud_shape1 = col1.selectbox("Cloud 1 Shape:", cloud_shape_choices)

                cloud_color2 = col2.selectbox("Cloud 2 Theme:", cloud_color_choices)
                cloud_font2 = col2.selectbox("Cloud 2 Font:", cloud_font_choices)
                cloud_bg2 = col2.selectbox("Cloud 2 Background:", cloud_bg_choices)
                cloud_shape2 = col2.selectbox("Cloud 2 Shape:", cloud_shape_choices)

                cloud_color3 = col3.selectbox("Cloud 3 Theme:", cloud_color_choices)
                cloud_font3 = col3.selectbox("Cloud 3 Font:", cloud_font_choices)
                cloud_bg3 = col3.selectbox("Cloud 3 Background:", cloud_bg_choices)
                cloud_shape3 = col3.selectbox("Cloud 3 Shape:", cloud_shape_choices)

                cloud_color4 = col4.selectbox("Cloud 4 Theme:", cloud_color_choices)
                cloud_font4 = col4.selectbox("Cloud 4 Font:", cloud_font_choices)
                cloud_bg4 = col4.selectbox("Cloud 4 Background:", cloud_bg_choices)
                cloud_shape4 = col4.selectbox("Cloud 4 Shape:", cloud_shape_choices)

                cloud_color5 = col5.selectbox("Cloud 5 Theme:", cloud_color_choices)
                cloud_font5 = col5.selectbox("Cloud 5 Font:", cloud_font_choices)
                cloud_bg5 = col5.selectbox("Cloud 5 Background:", cloud_bg_choices)
                cloud_shape5 = col5.selectbox("Cloud 5 Shape:", cloud_shape_choices)
                
                extra_stopwords = st.text_input("Remove stopwords (please separate words by comma):")
                
                submit_button = st.form_submit_button(label='Create Wordclouds')

            progress = st.header("Ready to plot! Click 'Create Wordcloud' to begin.")
            progressbar = st.progress(0)

        try:
            if submit_button:
                progress.header("Building wordclouds, please wait...")
                progressbar.progress(0.1)

                if datanumber == 2:
                    cloud_color_list = [cloud_color1, cloud_color2]
                    cloud_bg_list = [cloud_bg1, cloud_bg2]
                    cloud_shape_list = [cloud_shape1, cloud_shape2]
                    cloud_font_list = [cloud_font1, cloud_font2]

                if datanumber == 3:
                    cloud_color_list = [cloud_color1, cloud_color2, cloud_color3]
                    cloud_bg_list = [cloud_bg1, cloud_bg2, cloud_bg3]
                    cloud_shape_list = [cloud_shape1, cloud_shape2, cloud_shape3]
                    cloud_font_list = [cloud_font1, cloud_font2, cloud_font3]

                if datanumber == 4:
                    cloud_color_list = [cloud_color1, cloud_color2, cloud_color3, cloud_color4]
                    cloud_bg_list = [cloud_bg1, cloud_bg2, cloud_bg3, cloud_bg4]
                    cloud_shape_list = [cloud_shape1, cloud_shape2, cloud_shape3, cloud_shape4]
                    cloud_font_list = [cloud_font1, cloud_font2, cloud_font3, cloud_font4]

                if datanumber == 5:
                    cloud_color_list = [cloud_color1, cloud_color2, cloud_color3, cloud_color4, cloud_color5]
                    cloud_bg_list = [cloud_bg1, cloud_bg2, cloud_bg3, cloud_bg4, cloud_bg5]
                    cloud_shape_list = [cloud_shape1, cloud_shape2, cloud_shape3, cloud_shape4, cloud_shape5]
                    cloud_font_list = [cloud_font1, cloud_font2, cloud_font3, cloud_font4, cloud_font5]

                progressbar.progress(0.2)
                prepped_list = []
                progress_value = 0.2
                progress.header("Preprocessing...")
                increment = 0.1/datanumber

                for index in [0,1,2,3,4]:
                    
                    progress_value += increment 
                    progressbar.progress(progress_value)

                    if str(ss.df_list[index]) != "1":
                        if ss.prep_list[index] == 1:
                            ss.prep_list[index] = prep([ss.df_list[index]])
                            prepped_list.append(ss.prep_list[index])
                        else:
                            prepped_list.append(ss.prep_list[index])

                prepped_list = [item for sublist in prepped_list for item in sublist]
                progress.header("Vectorizing...")
                df = vectorize_multiple(prepped_list, extra_stopwords)
                total_dfs = len(prepped_list)
                wordcloud_list = []

                progress_value = 0.2
                for x in range(total_dfs):
                    progresstext = "Building wordcloud " + str(x+1) + " of " + str(total_dfs)
                    progress.header(progresstext)
                    increment = 0.6/total_dfs
                    progress_value += increment 
                    progressbar.progress(progress_value)
                    selected_df = df[[x]]
                    cloud_color = cloud_color_list[x]
                    cloud_bg = cloud_bg_list[x]
                    cloud_shape = cloud_shape_list[x]
                    cloud_font = cloud_font_list[x]
                    wordcloud = visualize(selected_df, cloud_color, cloud_bg, cloud_shape, cloud_font, x)
                    wordcloud_list.append(wordcloud)
                
                windows = total_dfs
                
                progressbar.progress(0.95)
                progress.header("Displaying wordclouds - this may take a second...")

                queries = [x for x in ss.query_list if x != 1]

                if windows == 2:
                    wc_col1, wc_col2 = st.beta_columns(windows)
                    wc_col1.subheader("*" + queries[0] + "*")
                    wc_col2.subheader("*" + queries[1] + "*")
                    wc_col1.image(wordcloud_list[0].to_array())
                    wc_col2.image(wordcloud_list[1].to_array())

                if windows == 3:
                    wc_col1, wc_col2, wc_col3 = st.beta_columns(windows)
                    wc_col1.subheader("*" + queries[0] + "*")
                    wc_col2.subheader("*" + queries[1] + "*")
                    wc_col3.subheader("*" + queries[2] + "*")
                    wc_col1.image(wordcloud_list[0].to_array())
                    wc_col2.image(wordcloud_list[1].to_array())
                    wc_col3.image(wordcloud_list[2].to_array())

                if windows == 4:
                    wc_col1, wc_col2 = st.beta_columns(2)
                    wc_col1.subheader("*" + queries[0] + "*")
                    wc_col2.subheader("*" + queries[1] + "*")
                    wc_col1.subheader("*" + queries[2] + "*")
                    wc_col2.subheader("*" + queries[3] + "*")
                    wc_col1.image(wordcloud_list[0].to_array())
                    wc_col2.image(wordcloud_list[1].to_array())
                    wc_col1.image(wordcloud_list[2].to_array())
                    wc_col2.image(wordcloud_list[3].to_array())

                if windows == 5:
                    wc_col1, wc_col2, wc_col3, wc_col4 = st.beta_columns([0.5,1,1,0.5])
                    wc_col1_2, wc_col2_2, wc_col3_3 = st.beta_columns(3)
                    wc_col2.subheader("*" + queries[0] + "*")
                    wc_col3.subheader("*" + queries[1] + "*")
                    wc_col1_2.subheader("*" + queries[2] + "*")
                    wc_col2_2.subheader("*" + queries[3] + "*")
                    wc_col3_3.subheader("*" + queries[4] + "*")
                    wc_col2.image(wordcloud_list[0].to_array())
                    wc_col3.image(wordcloud_list[1].to_array())
                    wc_col1_2.image(wordcloud_list[2].to_array())
                    wc_col2_2.image(wordcloud_list[3].to_array())
                    wc_col3_3.image(wordcloud_list[4].to_array())

                time.sleep(0.5)
                progressbar.progress(1.0)
                progress.header("Done! Save the wordclouds, or try changing the inputs for other results!")
        except AttributeError as e:
            st.write("Please scrape some data first, fool!")
            st.write("Error:", e)


def page_sentiment():
    st.header("**Sentiment Analysis of Comments**")
    st.write("Each comment is assigned sentiment scores weighted between 'positive', 'neutral', and 'negative' as well as a compound between them")
    nltk.download('vader_lexicon')
    sid = SentimentIntensityAnalyzer()
    #Generate sentiment scores
    #generate list of dataframes:
    list_dfs = []
    queries = []

    for index in [0,1,2,3,4]:
        if str(ss.df_list[index]) != "1":
            ss.df_list[index]['scores'] = ss.df_list[index]['Comment'].apply(lambda comment: sid.polarity_scores(comment))
            ss.df_list[index]['compound'] = ss.df_list[index]['scores'].apply(lambda x: x.get('compound'))
            ss.df_list[index]['positive'] = ss.df_list[index]['scores'].apply(lambda x: x.get('pos'))
            ss.df_list[index]['negative'] = ss.df_list[index]['scores'].apply(lambda x: x.get('neg'))
            ss.df_list[index]['neutral'] = ss.df_list[index]['scores'].apply(lambda x: x.get('neu'))

            list_dfs.append(ss.df_list[index])
            queries.append(ss.query_list[index])

    if len(queries) == 0:
            st.write("**You don't have any datasets loaded yet - analysis requires at least 1!**") 
            st.write("Please scrape or upload at least 1 dataset before continuing.")
    else:
        with st.beta_expander("What are sentiment scores?"):
                    st.markdown(
                        "Sentiment scores are generated using the VADER (Valence Aware Dictionary for Sentiment Reasoning)  \n"
                        "This model is sensitive to both polarity and intensity of emotion  \n"
                        "Each individual comment is assigned 4 scores based on their content:  \n   \n"
                        "⋅⋅⋅ **Positive**  \n"
                        "⋅⋅⋅ **Negative**  \n"
                        "⋅⋅⋅ **Neutral**  \n"
                        "⋅⋅⋅ **Compound**  \n  \n"
                        "The compound is the result of normalizing (between -1 and 1) the three other scores, providing a good estimate for overall sentence valence.  \n"
                        "Positive compound scores suggest positive comments, and vice versa" 
                    )

        formated = pd.concat(list_dfs)

        formated["Comment Published"] = pd.to_datetime(formated["Comment Published"], format= "%Y-%m-%dT%H:%M:%SZ")
        formated["Comment Published"] = pd.to_datetime(formated["Comment Published"], format='%Y-%m-%d')
        formated["Comment Published"] = formated["Comment Published"].dt.strftime('%Y-%m-%d')
        all_comments = formated.copy()
        all_comments = all_comments[["Query", "Comment", "Comment Published", 'compound', 'positive', 'negative','neutral']]
        formated = formated.groupby(['Query', 'Comment Published'], as_index=False)['compound', 'positive', 'negative','neutral'].mean()

        with st.form(key='my_form'):
            st.info("Below here, you can select which of the queries, you want to include in the sentiment chart.  \n"
            "The sentiment chart will plot the compound sentiment for all comments in each selected query over time  \n  \n"
            "The sentiment chart is **interactable**: On the timeline below the plot, you can select an interval to scale the plot to. This interval can be dragged across the timeline and the plot will adjust accordingly. The three dots in the top right corner, will allow you to download the plot as well as access the source code that generated your plot.")
            options = st.multiselect('Select data to include', queries, default = queries[0])

            submit_button = st.form_submit_button(label='Show sentiment')
            
            if submit_button:

                formated = formated[formated.Query.isin(list(options))]
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

        with st.beta_expander("See the aggregated sentiment compound scores"):
            st.info("Sentiment scores are mean aggregated by date and query. This means that for each of your YouTube scrapes, you get the compound sentiment over time for all comments captured by the query. "
            "Remember, positive compound scores reflect that the comments on that particular day primarily exhibited positive sentiment with higher scores reflecting increased sentiment strength. The opposite is true for negative values.   \n"
            "**Here is what that looks like:**")
            st.dataframe(formated)
            csv = formated.to_csv(index=False)
            b64 = base64.b64encode(csv.encode()).decode()  # some strings <-> bytes conversions necessary here
            link1 = f'<a href="data:file/csv;base64,{b64}" download="aggregated_sentiment.csv">Download CSV</a>'
            st.markdown(link1, unsafe_allow_html=True)
        
        with st.beta_expander("See the raw sentiment scores pr. comment"):
            st.info("These are the raw sentiment scores assigned to each comment. As sentiment is assigned to sentences as a whole instead of individual words, we can handle things as negations or strength markers.   \n   \n"
            "**Notice:** You can mouse hover over the individual comments to see the full text.")
            st.dataframe(all_comments)
            csv2 = all_comments.to_csv(index=False)
            b64_2 = base64.b64encode(csv2.encode()).decode()  # some strings <-> bytes conversions necessary here
            link2 = f'<a href="data:file/csv;base64,{b64_2}" download="raw_sentiment_comments.csv">Download CSV</a>'
            st.markdown(link2, unsafe_allow_html=True)


def page_topic():

    st.header("**Data Similarity Analysis**")
    names = [x for x in ss.query_list if x != 1]

    if len(names) < 2:
        st.write("**You don't have enough datasets loaded - PCA analysis requires at least 3!**") 
        st.write("Please scrape or upload at least 3 dataset before continuing.")
    else:
        st.info("Here, you can run a principal component analysis (PCA) on the TF-IDF matrix of the words in your scraped YouTube comments.   \n"
        "What this essentially does, is that it allows you assess, which of your different queries are similar or dissimilar to each other.   \n"
        "Do not worry - this application automatically handles the preprocessing and analysis, so you can focus on the results. Below here, you can read about how the analyses work and what they do.")

        with st.beta_expander("What is TF-IDF?"):
            st.markdown(
                "**Term frequency-inverse document frequency** (TF-IDF) is a statistical measure that evaluates how relevant a word is to a document in a collection of documents.   \n"
                "This is done by multiplying two metrics: how many times a word appears in a document, and the inverse document frequency of the word across a set of documents [(Ref)](https://monkeylearn.com/blog/what-is-tf-idf/)   \n   \n"
                "In this case, we are using TF-IDF to assess the importance of individual words of the comments in a youtube query relative to the other queries made.   \n"
                "The principal component analysis then uses this TF-IDF matrix to compare queries across all their word vectors. More on this can be found below."
            )
        with st.beta_expander("What is PCA?"):
            st.markdown(
                "**Principal component analysis** (PCA) is a dimensionality-reduction method that is often used to reduce the dimensionality of large data sets, by transforming a large set of variables into a smaller one that still contains most of the information in the large set [(Ref)](https://builtin.com/data-science/step-step-explanation-principal-component-analysis).    \n"
                "This has the merit of increasing interpretability but at the same time minimizing information loss. It does so by creating new uncorrelated variables that successively maximize variance [(Ref)](https://royalsocietypublishing.org/doi/10.1098/rsta.2015.0202).    \n   \n"
                "In this case, we are utilizing the TF-IDF matrix of vectors containing the relative importance of each word to each query. These word vectors each represent a dimension of word importance. The PCA effectively 'boils down' these dimensions to a 2-dimensional space. Each query can then be mapped into this new space and compared to other queries.    \n   \n"
                "**Notice** that the axis in this new 2-dimensional space no longer carry any semantic meaning, and interpretability is pointless."
                "The point of running PCA in this case, is that it allows us to see whether queries are similar to each other with regards to individual word importance."
                )

        col1, col2 = st.beta_columns([1,1])

        with col1.form(key='my_form'):
            extra_stopwords = st.text_input("Remove stopwords (please separate words by comma):")
            st.markdown("Here you can select specific words that won't be included in the principal companant analysis. The vectors for these specified words simply wont be included in the model.   \n   You can use the wordclouds to search for problematic keywords that you want to exclude.")
            st.info("**Please note**: Depending on the number of comments or loaded queries, this may take a few minutes. Please don't navigate to other dashboard tabs while this is running.")
            submit_button = st.form_submit_button(label='Run PCA')

        if submit_button:

            #Prep data:
            #collect all available data
            df_list = [x for x in ss.df_list if str(x) != str(1)]
            prepped_list = [] #empty placeholder

            #Check if data has already been preprocessed. If not - run preprocessing on each individual element

            for index in [0,1,2,3,4]:
                if str(ss.df_list[index]) != "1":
                    if ss.prep_list[index] == 1:
                        ss.prep_list[index] = prep([ss.df_list[index]])
                        prepped_list.append(ss.prep_list[index])
                    else:
                        prepped_list.append(ss.prep_list[index])

            #Unpack preprocessed data and prep query names
            prepped_list = [item for sublist in prepped_list for item in sublist]
            names = [x for x in ss.query_list if x != 1]
            dictOfWords = { i : names[i] for i in range(0, len(names) ) }
            
            #Run TF-IDF vectorizer and rename columns appropriately
            vectors = vectorize_pca(prepped_list, extra_stopwords)
            X = vectors.todense()

            #df = df.rename(columns = dictOfWords , inplace = False)

            num_clusters = len(df_list)
            num_seeds = len(df_list)
            max_iterations = 300
            labels_color_map = {
                0: '#20b2aa', 1: '#ff7373', 2: '#ffe4e1', 3: '#005073', 4: '#4d0404',
                5: '#ccc0ba', 6: '#4700f9', 7: '#f6f900', 8: '#00f91d', 9: '#da8c49'
            }
            pca_num_components = 2

            clustering_model = KMeans(
                n_clusters=num_clusters,
                max_iter=max_iterations,
                precompute_distances="auto",
                n_jobs=-1
            )

            labels = clustering_model.fit_predict(vectors)

            tags = names

            reduced_data = PCA(n_components=pca_num_components).fit_transform(X)
            # print reduced_data
            z = reduced_data[:,0]
            y = reduced_data[:,1]

            fig, ax = plt.subplots()
            for index, instance in enumerate(reduced_data):
                # print instance, index, labels[index]
                pca_comp_1, pca_comp_2 = reduced_data[index]
                color = labels_color_map[labels[index]]
                ax.scatter(pca_comp_1, pca_comp_2, c=color, s = 80)
                ax.axes.xaxis.set_visible(False)
                ax.axes.yaxis.set_visible(False)
            for i, txt in enumerate(tags):
                ax.annotate(txt, (z[i], y[i]))

            col1.write("Principal component analysis finished!")

            st.set_option('deprecation.showPyplotGlobalUse', False)
            col2.pyplot(fig, use_container_width=True )

        

            #gif_runner = st.image("rocket.gif")
            


#def page_about():
#    st.header("**Who Are We?**")
#    st.text("We made this, so we're pretty cool, hey.")   


if __name__ == "__main__":
    main()



