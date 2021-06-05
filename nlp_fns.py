from nltk.corpus import stopwords
import gensim
from gensim.utils import simple_preprocess
import spacy
import re
import os
import pickle
import pandas as pd
from pandas import DataFrame
# import google.oauth2.credentials
# from googleapiclient.discovery import build
# from googleapiclient.errors import HttpError
# from google_auth_oauthlib.flow import InstalledAppFlow
# from google.auth.transport.requests import Request
from youtube_transcript_api import YouTubeTranscriptApi
from sklearn.feature_extraction.text import TfidfVectorizer
import matplotlib.pyplot as plt

import numpy as np

from wordcloud import WordCloud
from PIL import Image
import streamlit as st

###############################################################################
###############################################################################

#### POSSIBLE FUNCTIONS TO MOVE ####

def pipeline_multiple(df_list, cloud_color_list, cloud_bg_list, cloud_shape_list, cloud_font_list, extra_stopwords):
    progress = st.text(
        '1/3: Running Term-Frequency Inverse-Document-Frequency analysis...')
    total_dfs = len(df_list)
    df_list = prep(df_list)
    progress.text('2/3: Vectorizing documents...')
    df = vectorize_multiple(df_list, extra_stopwords)
    progress.text('3/3: Building wordcloud visualization...')
    wordcloud_list = []
    for x in range(total_dfs):
        selected_df = df[[x]]
        cloud_color = cloud_color_list[x]
        cloud_bg = cloud_bg_list[x]
        cloud_shape = cloud_shape_list[x]
        cloud_font = cloud_font_list[x]
        wordcloud = visualize(selected_df, cloud_color,
                              cloud_bg, cloud_shape, cloud_font, x)
        wordcloud_list.append(wordcloud)
    progress.text('Done!')
    return wordcloud_list


def pipeline_single(df, cloud_color, cloud_bg, cloud_shape, cloud_font):
    progress = st.text('1/2: Vectorizing documents...')
 #   df = tfidf_prep(df)
  #  progress.text('3/4: Vectorizing documents...')
    df = vectorize_single(df)
    progress.text('2/2: Building wordcloud visualization...')
    wordcloud = visualize(df, cloud_color, cloud_bg, cloud_shape, cloud_font)
    progress.text('Done!')
    return wordcloud


###########################################################################

def sent_to_words(sentences):
    for sentence in sentences:
        # deacc=True removes punctuations
        yield(gensim.utils.simple_preprocess(str(sentence), deacc=True))


def lemmatization(texts, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV']):
    """https://spacy.io/api/annotation"""
    # Prepare stopwords and lemmatizer
    nlp = spacy.load('en_core_web_sm', disable=['parser', 'ner'])
    texts_out = []
    for sent in texts:
        doc = nlp(" ".join(sent))
        texts_out.append([token.lemma_ for token in doc])
    return texts_out


def make_bigrams(texts):
    return [bigram_mod[doc] for doc in texts]


def vectorize(df):
    print("Running TF-IDF on all data sets...")
    # prep stopwords
    more_stopwords = ("com, youtube, www, http, https").split(",")
    stops = list(stopwords.words('english'))
    stops.extend(more_stopwords)
    stops = [x.strip(' ') for x in stops]
    # , tokenizer=LemmaTokenizer())
    vectorizer = TfidfVectorizer(stop_words=stops, lowercase=True)
    vectors = vectorizer.fit_transform([df])
    feature_names = vectorizer.get_feature_names()
    dense = vectors.todense()
    denselist = dense.tolist()
    df = pd.DataFrame(denselist, columns=feature_names)
    # Clean results
    df = df.T
   # df = df.rename(columns = {0: 'word_word'}, inplace = False)
    return df


def visualize(df, cloud_color="gist_gray", cloud_bg="'Default (Transparent)'",  cloud_shape=None, cloud_font="AU", column=0):
    mode="RGB"

    if cloud_bg == 'Transparent':
        cloud_bg = "rgba(255, 255, 255, 0)"
        mode="RGBA"
    
    cloud_color = cloud_color.lower()

    cloud_bg = cloud_bg.lower()

    if cloud_shape == 'Square':
        cloud_shape = None

    # Scale word size mainly by rank (0) to frequency (1)
    cloud_scale = 0
    # Ratio of word angles from horizontal (1) to vertical (0)
    cloud_horizontal = 1

    if cloud_shape != None:
        cloud_shape = np.array(Image.open(
            "./resources/shapes/" + cloud_shape + ".png"))
    cloud_font = "./resources/fonts/" + cloud_font + ".ttf"
    keys = tuple(df.index.tolist())
    values = tuple(df[column])
    my_dict = dict(zip(keys, values))

    wordcloud = WordCloud(mode = mode, font_path=cloud_font, mask=cloud_shape, regexp=None, relative_scaling=cloud_scale, prefer_horizontal=cloud_horizontal,
                          width=800, height=800, background_color=cloud_bg, max_words=500, contour_width=0, colormap=cloud_color)

    wordcloud.generate_from_frequencies(my_dict)

    return wordcloud


def vectorize_single(document):

    more_stopwords = ("com, youtube, www, http, https").split(",")
    stops = list(stopwords.words('english'))
    stops.extend(more_stopwords)
    stops = [x.strip(' ') for x in stops]

    # extract comments
    document = document.Comment.values.tolist()
    # Remove mess
    document = [re.sub('\S*@\S*\s?', '', sent) for sent in document]
    # Remove new line characters
    document = [re.sub('\s+', ' ', sent) for sent in document]
    # Remove distracting single quotes
    document = [re.sub("\'", "", sent) for sent in document]
    # run tokenization:
    document = list(sent_to_words(document))
    # Run stopword removal:
    document = [[word for word in simple_preprocess(
        str(doc)) if word not in set(stops)] for doc in document]
    # Run lemmatization:
########  document = lemmatization(document, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV'])
    # Remove weird tags:
    document = [[word for word in simple_preprocess(
        str(doc)) if word not in set(stops)] for doc in document]
    # Bigrams:
  #  bigram = gensim.models.Phrases(document, min_count=10, threshold=30)
  #  bigram_mod = gensim.models.phrases.Phraser(bigram)
  #  document = make_bigrams(document)
    # Smash to string:
    document = [' '.join(sent) for sent in document]
    document = ' '.join(document)

    # Vectorizing
    # , tokenizer=LemmaTokenizer())
    vectorizer = TfidfVectorizer(stop_words=stops, lowercase=True)
    vectors = vectorizer.fit_transform([document])
    feature_names = vectorizer.get_feature_names()
    dense = vectors.todense()
    denselist = dense.tolist()
    document = pd.DataFrame(denselist, columns=feature_names)
    # Clean results
    document = document.T
  #  document = document.rename(columns = {0: 'word_word'}, inplace = False)
    return document

def vectorize_pca(df_list, extra_stopwords):
    # prep stopwords
    stops = list(stopwords.words('english'))
    extra_stopwords = extra_stopwords.split(",")
    stops.extend(extra_stopwords)
    stops = [x.strip(' ') for x in stops]
    # , tokenizer=LemmaTokenizer())
    vectorizer = TfidfVectorizer(stop_words=stops, lowercase=True)
    if len(df_list) == 1:
        df_list = df_list[0]
    else:
        df_list = df_list
    vectors = vectorizer.fit_transform(df_list)

    return vectors

def vectorize_multiple(df_list, extra_stopwords):
    print("Running TF-IDF on all data sets...")
    # prep stopwords
    stops = list(stopwords.words('english'))
    extra_stopwords = extra_stopwords.split(",")
    stops.extend(extra_stopwords)
    stops = [x.strip(' ') for x in stops]
    # , tokenizer=LemmaTokenizer())
    vectorizer = TfidfVectorizer(stop_words=stops, lowercase=True)
    if len(df_list) == 1:
        df_list = df_list[0]
    else:
        df_list = df_list
    vectors = vectorizer.fit_transform(df_list)

    feature_names = vectorizer.get_feature_names()
    dense = vectors.todense()
    denselist = dense.tolist()
    df = pd.DataFrame(denselist, columns=feature_names)
    # Clean results
    df = df.T
  #  df = df.rename(columns = {0: 'doc_1', 1: 'doc_2'}, inplace = False)
    return df


def prep(df_list):
    new_list = []
    for document in df_list:
        def make_bigrams(texts):
            return [bigram_mod[doc] for doc in texts]
        # prep stopwords
        more_stopwords = ("com, youtube, www, http, https").split(",")
        stops = list(stopwords.words('english'))
        stops.extend(more_stopwords)
        stops = [x.strip(' ') for x in stops]

        # extract comments
        document = document.Comment.values.tolist()
        # Remove mess
        document = [re.sub('\S*@\S*\s?', '', sent) for sent in document]
        # Remove new line characters
        document = [re.sub('\s+', ' ', sent) for sent in document]
        # Remove distracting single quotes
        #document = [re.sub("\'", "", sent) for sent in document]
        # run tokenization:
        document = list(sent_to_words(document))
        # Run stopword removal:
        document = [[word for word in simple_preprocess(
            str(doc)) if word not in set(stops)] for doc in document]
        # Run lemmatization:
    ###    document = lemmatization(document, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV'])
        # Remove weird tags:
        #document = [[word for word in simple_preprocess(str(doc)) if word not in set(stops)] for doc in document]
        # Bigrams:
        bigram = gensim.models.Phrases(document, min_count=5, threshold=10)
        bigram_mod = gensim.models.phrases.Phraser(bigram)
        document = make_bigrams(document)
        # Smash to string:
        document = [' '.join(sent) for sent in document]
        document = ' '.join(document)

        new_list.append(document)
    return new_list
