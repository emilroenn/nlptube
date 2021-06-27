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

    sidebar_storage = st.sidebar.beta_expander("Data Storage", expanded = True)
    with sidebar_storage:
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
    st.subheader("**A fast and easy scraping and analysis tool for the YouTube API!**")
    st.write("Ever wondered what YouTube's users think about funny cat videos? Which feelings they express about politics, social issues, or the latest fads? Or are you just curious to explore what a splash of big data and a couple of natural language processing tools can come up with? Then you've come to the right place!")
    st.write("**NLP and Big Data at your fingertips!**")
    st.write("Big data scraping and analysis can be a real hassle, from setting up API credentials and access tokens to complex data wrangling and preprocessing pipelines. Having been through the process ourselves, we know all too well how much time is spent setting up the infrastructure before finally being able to get to the insights.  \n\n That's why we decided to find a way to let others skip the boring parts and get straight to the results! The YouNLP app provides easy access to YouTube data scraping and language analysis through a series of easy-to-use tools. Simply choose what kind of data you'd like to look at, and what kind of analysis tools to try out - the rest is handled by the app!")

    col1,col2 = st.beta_columns([2,1])
    with col1:
        st.write("**Get smart!**")
        st.write("Want to have a quick overview of the tools and methods available? Head on over to the *How Does It Work?* page under the *About This App* menu in the sidebar.")
        st.write("**Get data!**")
        st.write("Before diving into analysis, you have to set up some data first! Head over to the *Manage Data* menu in the sidebar: here, you can scrape a new dataset on the *Scrape YouTube Data* page, upload data from previous sessions on the *Upload YouTube Data* page, or load in some of our sample datasets on the *Manage Stored Data* page! Up to 5 datasets can be stored in the app simultaneously, and can be independently explored or compared with the available tools!")
        st.write("**Get results!**")
        st.write("Once you've acquired your data, head on over to the *Analyze Data* menu! Here, you can use a variety of tools to explore and compare your datasets. Whether it's visualizing your data in customizable wordclouds, plotting the various sentiments of your data over time, or examining the similarities between your chosen datasets, the tools are quick and easy to use.")
        st.write("")
        st.write("Happy analyzing!")
        st.write("*- Emil & Lasse*")
    with col2:
        st.image("./resources/screenshots/flowchart_main.png")


def page_how():
    st.header('**How To Use This Tool**')
    col1, col2 = st.beta_columns([2,1])
    col1.write("YouNLP is a plug-and-play type analysis tool. Under normal circumstances, natural language processing analysis of social media texts such as YouTube comments takes a lot of time and effort. Scraping commments from YouTube requires users to know both programming and the YouTube API system, as well as create an authorized account for accessing the website's systems. Once the data is collected, it has to go through several steps of cleaning and preprocessing before it is useful to analysis tools.")
    col1.write("Because we know that struggle all too well ourselves, we decided to do the hard work for you. Behind the scenes, the YouNLP application is already set up to access the YouTube API, scrape comments, process them, and run them through various types of analysis pipelines. All you have to do, is to put in your search term, and let the tool do the rest for you.")
    col1.write("For an overview of the process, see the flowchart on the right. If you're interested in any of the involved processes, feel free to read more about them in the dropdown menus below!")
    col2.image("./resources/screenshots/flowchart.png")

    st.subheader("**Manage Data**")
    st.write("The *Manage Data* menu of the sidebar allows you to scrape new data from YouTube, upload previously scraped data, and view, delete, or load sample data into the app's data containers.")
    st.info("Click one of the drop-down menus below to learn more about the tool in the Manage Data menu!")
    
    with st.beta_expander("Scrape YouTube Data", expanded = False):
        col1,col2 = st.beta_columns([2,1])
        col1.subheader("*Scrape YouTube Data*")
        col1.write("The *Scrape YouTube Data* page allows you to create a new dataset for analysis based on any search term. The tool's menu provides you with 3 options: 1) a term to search for, 2) number of videos to gather data from, and 3) an app data container to store the scraped data in.")
        col1.write("Once you start the scraper using the button, YouNLP accesses YouTube's API, plugs the search term into YouTube's search function, and returns a list of the top videos from the search. The app then sends a request to YouTube's API to return up to the top 100 comments from each listed video.")
        col1.write("These comments are then downloaded, along with other information such as each comment's publishing time and likes, as well as the video's title, views, channel, and more. Finally, the information is converted to a large dataframe and stored in the app's data container for further analysis - or for you to download.")
        col2.image("./resources/screenshots/scrape.png")

    with st.beta_expander("Upload YouTube Data", expanded = False):
        col1,col2 = st.beta_columns([2,1])
        col1.subheader("*Upload YouTube Data*")
        col1.write("The *Upload YouTube Data* page allows you to upload an existing dataset CSV file to the YouNLP app. As YouNLP only stores data for as long as the app is open in your browswer, this makes it easier to scrape and store multiple datasets over time, or return at a later time for analysis or novel comparisons.")
        col1.write("Be aware that the YouNLP app is currently only able to handle datasets of the same format as the one created by the app. This also means, that if you change any formatting or overwrite the original CSV-file after opening it (e.g. by examining it in Excel), the file may not be recognized by the app.")
        col2.image("./resources/screenshots/upload.png")

    with st.beta_expander("Manage Stored Data", expanded = False):
        col1,col2 = st.beta_columns([2,1])
        col1.subheader("*Manage Stored Data*")
        col1.write("The *Manage Stored Data* page allows you to get an overview of and manage datasets currently stored in the app's containers. The page provides a live view of the app's 5 data containers, their contents, size, and search queries, and allows you to download or remove the datasets currently stored.")
        col1.write("For the sake of exploration, we have also added a small set of sample datasets for you to try out the app with. Simple select a container and a dataset to load into it, and hit the button to load it up!")
        col1.write("Keep in mind that only one dataset can be stored in each container, and that datasets not downloaded before overwriting or emptying their containers are permanently lost.")
        col2.image("./resources/screenshots/manage.png")

    st.subheader("**Analyze Data**")
    st.write("The *Analyze Data* menu of the sidebar provides a variety of easy-to-use language analysis tools.")
    st.info("Click one of the drop-down menus below to learn more about the tools in the Analyze Data menu!")
    
    with st.beta_expander("Wordcloud Analysis", expanded = False):
        col1,col2 = st.beta_columns([2,1])
        col1.subheader("*Wordcloud Analysis*")
        col1.write("The *Wordcloud Analysis* page allows you to get a quick overview of your data using one of the oldest tricks in the book - a wordcloud! Wordclouds are visual representations of the most commonly found words in a text, usually arranged in various shapes and colors, and give an intuitive overview of the kind of language used within the text.")
        col1.write("Wordclouds can be created using different techniques and from different types of data analysis. YouNLP allows for two different versions: frequency-based wordclouds of individual datasets, and TF-IDF-score-based wordclouds from multiple datasets. These can both be accessed on the top of the Wordcloud Analysis page.")
        col1.write("**Individual wordclouds**")
        col1.write("The Individual Datasets subpage allows you to visualize wordclouds from single datasets. Here, you can choose one of the datasets currently stored in the app, and use the tool to see which words most frequently occur in that data. The size of each word corresponds to its frequency within the data, with more common words shown as larger, and less common words as smaller.")
        col2.image("./resources/screenshots/wc_single.png")
        col1.write("**Multiple wordcloud comparison**")
        col1.write("The Compare Datasets subpage allows you to compare wordclouds across multiple datasets. Instead of simply showing the most common words in each dataset however, word sizes are based on TF-IDF scores for each word. In short, this means that the size of each word corresponds to how uniquely important that word is to the dataset. If a word appears often in all compared datasets, it therefore becomes smaller, as it is not unique to any of them. Likewise, if a word is used relatively often in one dataset, but not at all in the others, it is shown as larger.")
        col2.image("./resources/screenshots/wc_multiple.png")
        col1.write("**Customize your clouds**")
        col1.write("While the tool automatically removes a list of common English-language stopwords such as 'the' or 'it', you can always add additional words to be excluded from the wordclouds. You are also free to customize your wordsclouds with different colors, fonts, or shapes - or simply run the tool again for new cloud configurations!")


    with st.beta_expander("Sentiment Analysis", expanded = False):
        col1,col2 = st.beta_columns([2,1])
        col1.subheader("*Sentiment Analysis*")
        col1.write("One interesting avenue of exploration in the acquired comments is the sentiment expressed across your search queries. The **sentiment analysis tool** allows you to do just that!" )
        col1.write("Start by selecting the queries of your loaded datascrapes that you want to be included in the analysis. Then press the **Show sentiment** button and we will handle the rest. This tool automatically assigns a sentiment score to all selected comments.")
        col1.write("This processing should run fairly quickly. Upon finishing, You will have access to the raw dataframe with comments and their assigned sentiment scores. Moreover, these sentiment scores are also aggregated across your selected queries and by date, and shown in an interactive chart, allowing you to assess how the sentiment has developed over time. These aggregated scores are also available for download, so that you can use them for your own interesting investigations.")
        col2.image("./resources/screenshots/sentiment.png")

    with st.beta_expander("Similarity Analysis", expanded = False):
        col1,col2 = st.beta_columns([2,1])
        col1.subheader("*Similarity Analysis*")
        col1.write("Short info text.")
        col2.image("./resources/screenshots/pca.png")



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
        col3.write("**To analyse data, you gotta have data first!**  \n This tool executes a YouTube search for an input query, finds the number of videos selected, and scrapes up to the top 100 comments of each video. Details about the scraped videos and comments are then converted to a data frame and stored in the selected container in the app.  \n\n After scraping is complete, feel free to download the data frame or use one of the tools in the sidebar for further analysis. You can store up to 5 datasets in the app's containers.")
        containers = ['Container 1', 'Container 2', 'Container 3', 'Container 4', 'Container 5']
        dataset = col1.radio("Stored scraped data in:", (containers))
        df_index = int(dataset[-1])-1
        st.write("*Please note that YouTube's API currently imposes a few limitations on the scraping tool. A maximum of 50 videos can be requested per search, and a maximum of 100 searches may be requested per day across the application.*")
        st.info("ATTENTION: If the selected data container already has data stored, this data will be permanently overwritten. If you wish to keep the data, please select another container above or download the data from the sidebar before continuing.")
        user_input1 = col2.text_input("Search term:", '')
        user_input2 = col2.number_input("Videos:", min_value = 1, max_value = 50, value = 10)
      #  st.info("Please note that YouTube's API currently imposes a few limitations on the scraping tool. A maximum of 50 videos can be requested per search, and a maximum of 100 searches may be requested per day across the application.")
        submit_button = st.form_submit_button(label='Start Scraper')

    if submit_button:
        if user_input1 == "":
            st.info("Search term is missing - please enter a search term before scraping!")
        if user_input2 > 50:
            st.info("Maximum videos is 50 - please enter a new value before scraping!")
        if (user_input1 != "" and user_input2 < 51):
            try:
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

            except PermissionError:
                progress.subheader("Scraping cancelled: Couldn't access the YouTube API  : (")
                st.info("Whoops - looks like YouTube isn't cooperating at the moment. While we fix the problem, feel free to try out YouNLP's analysis tools on one of our sample datasets available on the *Manage Stored Data* page!")


def page_manage():
    st.header('**Manage Datasets**')


    st.info("Here, you can examine, download, delete, or load in sample data to the app's containers. Please note that once a container is emptied or overwritten, the data previously stored in that container is permanently deleted. Using the links under each container, you can download datasets and later reupload them for additional analysis or comparison.")
    st.subheader("Container Status")
    with st.form(key='my_form'):
    
        col0, col1, col2, col3, col4, col5 = st.beta_columns(6)
        containers = ['Container 1', 'Container 2', 'Container 3', 'Container 4', 'Container 5']
        dataset = col0.radio("Select container to empty:", (containers)) 
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


    st.subheader("Load Sample Data")
    with st.form(key='my_form2'):
        subcol1, subcol2, subcol3 = st.beta_columns([1,1,2])
        
        containers = ['Container 1', 'Container 2', 'Container 3', 'Container 4', 'Container 5']
        testsets = ['Dogecoin', 'Bitcoin', 'Biden', 'Trump', 'NLP']
        testdata = subcol2.radio("Choose a sample dataset:", (testsets)) 
        dataset = subcol1.radio("Select container to upload sample data:", (containers)) 
        df_index = int(dataset[-1])-1

        subcol3.write("**Test out the app without scraping!**")
        subcol3.write("If you want to take the various tools out for a spin without going through the scraping or uploading process, try loading in one of our existing data samples for exploration.  \n These datasets have been meticulously curated for the linguistic connoisseur, giving exciting insights into interesting topics such as memes, cryptocurrencies, and language analytics!")
        
        submit_button2 = st.form_submit_button(label='Load Test Data')

        
    if submit_button2:
        if str(ss.df_list[df_index]) != str(1):
            st.info("Container is already in use! To avoid accidental overwriting, please choose another container or empty the selected container.")
        else:
            file = "./resources/testdata/" + testdata + ".csv"
            ss.df_list[df_index] = pd.read_csv(file)
            ss.prep_list[df_index] = 1
            ss.query_list[df_index] = ss.df_list[df_index].at[2,'Query']
            ss.length_list[df_index] = len(ss.df_list[df_index])
            csv = ss.df_list[df_index].to_csv(index=False)
            b64 = base64.b64encode(csv.encode()).decode()  # some strings <-> bytes conversions necessary here
            ss.href_list[df_index] = f'<a href="data:file/csv;base64,{b64}" download="{ss.query_list[df_index]}.csv">Download CSV</a>'
            st.info(f"File using the query '{ss.query_list[df_index]}' with {ss.length_list[df_index]} comments successfully loaded into Data Container {df_index+1}")

    st.subheader("Examine Stored Datasets")

    for index in [0,1,2,3,4]:
        if str(ss.df_list[index]) != "1":
            with st.beta_expander("Examine data in Container " + str(index+1), expanded = False):
                st.dataframe(ss.df_list[index].head(10))
    

    for index, col in zip([0,1,2,3,4], [col1, col2, col3, col4, col5]):
        if str(ss.df_list[index]) == "1":
            col.markdown('<font color=grey>**CONTAINER ' + str(index+1) + ':** \n *Not in use*</font>', unsafe_allow_html=True)
        else:
            col.markdown('<font color=green>**CONTAINER ' + str(index+1) + ':**</font>', unsafe_allow_html=True)
            datatext = "**Search term: **" + str(ss.query_list[index]) + "  \n   **Comments:** " + str(ss.length_list[index])
            col.write(datatext)
            col.markdown(ss.href_list[index], unsafe_allow_html=True)



def page_visualize():

    cloud_color_choices = ['Summer','Autumn', 'Winter', 'Spring', 'Gray']
    cloud_bg_choices = ['Transparent','Black', 'White', 'Gray', 'Beige']
    cloud_font_choices = ['AU Passata','Spicy Rice', 'Raleway', 'Kenyan Coffee', 'Comic Sans']
    cloud_shape_choices = ['Square','Circle', 'Heart', 'Brain', 'Mushroom']

    st.header('**Wordcloud Visualization and Comparison**')
    col1, col2 = st.beta_columns(2)
    type = col1.selectbox(label = "",options = ['Individual Datasets (Word Frequency)', 'Compare Datasets (TF-IDF Scores)'])

    st.info("Use the menu above to switch between wordclouds examining individual datasets (using word frequency) and wordclouds comparing multiple datasets (using TF-IDF scores). For more information on both, see the *How Does It Work?* page.")   

    if type == "Individual Datasets (Word Frequency)":
        datanumber = 0
        for index in [0,1,2,3,4]:
            if str(ss.df_list[index]) != str(1):
                datanumber += 1


        if datanumber == 0: 
            st.write("**You don't have any datasets stored yet - this analysis requires at least 1!**") 
            st.info("To use this tool, please head over to the **Manage Data** menu in the sidebar. There, you can *create a new dataset* on the **Scrape YouTube Data** page of the menu, *upload a dataset* from a previous search on the **Upload YouTube Data** page, or *load a sample dataset* from the **Manage Stored Data** page!")

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

                col4.write("**They're clouds, but made from words!**  \n Wordclouds are one of the most simple yet effective visualizations of large amounts of text data. Wordclouds are visual representations of text data, showing the words most commonly occuring in the dataset.")   
                col1, col2 = st.beta_columns([4,2])
                extra_stopwords = col1.text_input("Remove stopwords (please separate words by comma):")
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
            st.write("**You don't have any datasets stored yet - this analysis requires at least 2!**") 
            st.info("To use this tool, please head over to the **Manage Data** menu in the sidebar. There, you can *create a new dataset* on the **Scrape YouTube Data** page of the menu, *upload a dataset* from a previous search on the **Upload YouTube Data** page, or *load a sample dataset* from the **Manage Stored Data** page!")

        if datanumber == 1:
            st.write("**You only have 1 dataset stored - this analysis requires at least 2!**") 
            st.info("To use this tool, please head over to the **Manage Data** menu in the sidebar. There, you can *create a new dataset* on the **Scrape YouTube Data** page of the menu, *upload a dataset* from a previous search on the **Upload YouTube Data** page, or *load a sample dataset* from the **Manage Stored Data** page!")

        if datanumber == 2:
            st.info("**2 datasets found in storage. Customize and create wordclouds below to compare them!**")
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
            st.info("**3 datasets found in storage. Customize and create wordclouds below to compare them!**")
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
            st.write("**4 datasets found in storage. Customize and create wordclouds below to compare them!**")
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
            st.write("**5 datasets found in storage. Customize and create wordclouds below to compare them!**")
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
            st.write("**You don't have any datasets stored yet - this analysis requires at least 1!**") 
            st.info("To use this tool, please head over to the **Manage Data** menu in the sidebar. There, you can *create a new dataset* on the **Scrape YouTube Data** page of the menu, *upload a dataset* from a previous search on the **Upload YouTube Data** page, or *load a sample dataset* from the **Manage Stored Data** page!")

    else:
        st.write("Each comment is assigned sentiment scores weighted between 'positive', 'neutral', and 'negative' as well as a compound between them")

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

    if len(names) < 3:
        st.write("**You don't have any datasets stored yet - this analysis requires at least 3!**") 
        st.info("To use this tool, please head over to the **Manage Data** menu in the sidebar. There, you can *create a new dataset* on the **Scrape YouTube Data** page of the menu, *upload a dataset* from a previous search on the **Upload YouTube Data** page, or *load a sample dataset* from the **Manage Stored Data** page!")

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



