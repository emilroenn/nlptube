import streamlit as st

global total_scraped
total_scraped = 0

st.title('Fucking Fancy Youtube Scraper (TM)')

user_input1 = st.text_input("Search term:", '')
user_input2 = st.number_input("Number of results:", 1)

data_load_state = st.text('Loading script, please wait...')
scrape_state = st.text(('Total videos scraped: ' + str(total_scraped)+". Total scrapes remaining: approximately " + str(10000-total_scraped)))


###############################

API_key = "AIzaSyA_OuQywrdv4MSgel1xFnnAvdgR0zvR58M"
CLIENT_SECRETS_FILE = "client_secret.json"

SCOPES = ['https://www.googleapis.com/auth/youtube.force-ssl']
API_SERVICE_NAME = 'youtube'
API_VERSION = 'v3'

code = "4/1AfDhmrgsalD6LJlRuWliCzO0o7o8qLouK6Lg3plZMCW2H5dyeDAYUGrdfGY"

from nltk.corpus import stopwords
import gensim
from gensim.utils import simple_preprocess
import spacy
import re
import os
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



def get_authenticated_service():
    credentials = None
    if os.path.exists('token.pickle'):
        with open('token.pickle', 'rb') as token:
            credentials = pickle.load(token)
    if not credentials or not credentials.valid:
        if credentials and credentials.expired and credentials.refresh_token:
            credentials.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(
                CLIENT_SECRETS_FILE, SCOPES)
            credentials = flow.run_console()
        with open('token.pickle', 'wb') as token:
            pickle.dump(credentials, token)
    return build(API_SERVICE_NAME, API_VERSION, credentials = credentials)

if __name__ == '__main__':
    os.environ['OAUTHLIB_INSECURE_TRANSPORT'] = '1'
    service = get_authenticated_service()


def get_data(query, results):

    # =============================================================================
    # Search Query Initialisation
    # =============================================================================

    query_results = service.search().list(
            part = 'snippet',
            q = query,
            order = 'relevance', # You can consider using viewCount
            maxResults = results,
            type = 'video', # Channels might appear in search results
            relevanceLanguage = 'en',
            safeSearch = 'moderate',
            ).execute()

    # =============================================================================
    # Get Video IDs
    # =============================================================================
    video_id = []
    channel = []
    video_title = []
    video_desc = []
    video_published = []
    for item in query_results['items']:
        video_id.append(item['id']['videoId'])
        channel.append(item['snippet']['channelTitle'])
        video_title.append(item['snippet']['title'])
        video_desc.append(item['snippet']['description'])
        video_published.append(item['snippet']['publishedAt'])


    # =============================================================================
    # Get Comments of Top Videos
    # =============================================================================
    video_id_pop = []
    channel_pop = []
    video_title_pop = []
    video_desc_pop = []
    comments_pop = []
    comment_id_pop = []
    reply_count_pop = []
    like_count_pop = []
    comment_published_pop = []
    video_published_pop = []

    errors = 0
    progress = 0
    progressbar = st.progress(0)

    from tqdm import tqdm
    for i, video in enumerate(tqdm(video_id, ncols = 100)):
        progress = progress + 1
        progress_number = int(progress/results*100)
        progressbar.progress(progress_number)
        global total_scraped
        total_scraped = total_scraped + 1
        try:
            response = service.commentThreads().list(
                            part = 'snippet',
                            videoId = video,
                            maxResults = 100, # Only take top 100 comments...
                            order = 'relevance', #... ranked on relevance
                            textFormat = 'plainText',
                            ).execute()
            
            comments_temp = []
            comment_id_temp = []
            reply_count_temp = []
            like_count_temp = []
            comment_published_temp = []

            for item in response['items']:
                comments_temp.append(item['snippet']['topLevelComment']['snippet']['textDisplay'])
                comment_id_temp.append(item['snippet']['topLevelComment']['id'])
                reply_count_temp.append(item['snippet']['totalReplyCount'])
                like_count_temp.append(item['snippet']['topLevelComment']['snippet']['likeCount'])
                comment_published_temp.append(item['snippet']['topLevelComment']['snippet']['updatedAt'])
            
            comments_pop.extend(comments_temp)
            comment_id_pop.extend(comment_id_temp)
            reply_count_pop.extend(reply_count_temp)
            like_count_pop.extend(like_count_temp)
            comment_published_pop.extend(comment_published_temp)

            video_id_pop.extend([video_id[i]]*len(comments_temp))
            channel_pop.extend([channel[i]]*len(comments_temp))
            video_title_pop.extend([video_title[i]]*len(comments_temp))
            video_desc_pop.extend([video_desc[i]]*len(comments_temp))
            video_published_pop.extend([video_published[i]]*len(comments_temp))
        except Exception as e:
            #print("Error:",e)
            errors = errors + 1
            pass

    query_pop = [query] * len(video_id_pop)

    print("Total errors:", errors)


    # =============================================================================
    # Populate to Dataframe
    # =============================================================================

    output_dict = {
            'Query': query_pop,
            'Channel': channel_pop,
            'Video Title': video_title_pop,
            'Video Description': video_desc_pop,
            'Video ID': video_id_pop,
            'Comment': comments_pop,
            'Comment ID': comment_id_pop,
            'Replies': reply_count_pop,
            'Likes': like_count_pop,
            'Video Published': video_published_pop,
            'Comment Published': comment_published_pop
            }

    output_df = pd.DataFrame(output_dict, columns = output_dict.keys())

    return output_df


###############################################################################
###############################################################################


def sent_to_words(sentences):
    for sentence in sentences:
        yield(gensim.utils.simple_preprocess(str(sentence), deacc=True)) # deacc=True removes punctuations

def lemmatization(texts, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV']):
    """https://spacy.io/api/annotation"""
    texts_out = []
    for sent in texts:
        doc = nlp(" ".join(sent)) 
        texts_out.append([token.lemma_ for token in doc])
    return texts_out

def make_bigrams(texts):
    return [bigram_mod[doc] for doc in texts]

def tfidf_prep(document):
    def make_bigrams(texts):
        return [bigram_mod[doc] for doc in texts]
    #select drug
    #extract comments
    document = document.Comment.values.tolist()
    #Remove mess
    document = [re.sub('\S*@\S*\s?', '', sent) for sent in document]
    # Remove new line characters
    document = [re.sub('\s+', ' ', sent) for sent in document]
    # Remove distracting single quotes
    document = [re.sub("\'", "", sent) for sent in document]
    #run tokenization:
    document = list(sent_to_words(document))
    #Run stopword removal:
    document = [[word for word in simple_preprocess(str(doc)) if word not in set(stops)] for doc in document]
    #Run lemmatization:
    document = lemmatization(document, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV'])
    #Remove weird tags:
    document = [[word for word in simple_preprocess(str(doc)) if word not in set(stops)] for doc in document]
    #Bigrams:
    bigram = gensim.models.Phrases(document, min_count=10, threshold=30)
    bigram_mod = gensim.models.phrases.Phraser(bigram)
    document = make_bigrams(document)
    #Smash to string:
    document = [' '.join(sent) for sent in document]
    document = ' '.join(document)
    return document


# Prepare stopwords and lemmatizer
print("Preparing stopwords and lemmatizer...")
nlp = spacy.load('en_core_web_sm', disable=['parser', 'ner'])
#nlp.max_length = 100000000
more_stopwords = ("de, much, dont, im, liking, getting, gotten, ever, always, every, many, few, even, got, get, make, made, let, makes, im, say, said, better, best, really, would, something, came,much, oh, put, also, lyrics, lyric, comment, comments, lot, 2020, 2019, 2018, hey, hi, may, making, many, cause,com, youtube, etc, else, since, comes, come, www, la,que, re, el, use, used, still, much, ever, every, around,also, da, finally, sure, literal, literally, ooh, ya, every, album, let, -PRON-, pron, http, https").split(",")
stops = list(stopwords.words('english'))
stops.extend(more_stopwords)
stops = [x.strip(' ') for x in stops]


def vectorize(df):
    print("Running TF-IDF on all data sets...")
    vectorizer = TfidfVectorizer(stop_words = stops, lowercase = True)#, tokenizer=LemmaTokenizer())
    vectors = vectorizer.fit_transform([df])
    feature_names = vectorizer.get_feature_names()
    dense = vectors.todense()
    denselist = dense.tolist()
    df = pd.DataFrame(denselist, columns=feature_names)
    #Clean results
    df = df.T
    df = df.rename(columns = {0: 'Psychedelics'}, inplace = False)
    return df

def visualize(df):
    cloud_color = "summer"     #summer # Set wordcloud color scheme ('Wistia' or 'OrRd' are nice! YlGn is great for ayahuasca!)
    cloud_bg_color = "black"    # Set background color ("#182a2a" is a nice, dark green; #2b2d2f coal grey!)
    cloud_scale = 0             # Scale word size mainly by rank (0) to frequency (1) 
    cloud_horizontal = 1        # Ratio of word angles from horizontal (1) to vertical (0) 
    keys = tuple(df.index.tolist())
    values = tuple(df['Psychedelics'])
    my_dict = dict(zip(keys,values))

    wordcloud = WordCloud(regexp=None, relative_scaling=cloud_scale, prefer_horizontal=cloud_horizontal, width = 500 , height = 500, background_color=cloud_bg_color, max_words=500, contour_width=0, colormap = cloud_color)

    wordcloud.generate_from_frequencies(my_dict)

    return wordcloud



def pipeline1 (query,results):
    if query == "":
        return "Awaiting input"
    else:
        data_load_state.text('1/4: Scraping YouTube...')
        df = get_data(query, results)
        data_load_state.text('2/4: Running Term-Frequency Inverse-Document-Frequency analysis...')
        df = tfidf_prep(df)
        data_load_state.text('3/4: Vectorizing documents...')
        df = vectorize(df)
        return df


def pipeline2 (df):
    if str(df) == "Awaiting input":
        return "Awaiting input"
    else:
        data_load_state.text('4/4: Building wordcloud visualization...')
        wordcloud = visualize(df)
        return wordcloud



data_load_state.text('Script loaded! Awaiting input...')

if st.button('Scrape YouTube'):
    data = pipeline1(user_input1,int(user_input2))

if st.button('Create Wordcloud'):
    wordcloud = pipeline2(data)

#if wordcloud != "Awaiting input":
#    st.image(wordcloud.to_array())
#    data_load_state.text('Done! Awaiting next input...')
#if wordcloud == "Awaiting input":
#    data_load_state.text('Script loaded! Awaiting input...')

scrape_state.text(('Total videos scraped: ' + str(total_scraped)+". Total scrapes remaining: approximately " + str(10000-total_scraped)))
