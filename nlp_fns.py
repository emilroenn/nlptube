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
from nlp_fns import * 


from stqdm import stqdm
from wordcloud import WordCloud
from PIL import Image

###############################################################################
###############################################################################

def sent_to_words(sentences):
    for sentence in sentences:
        yield(gensim.utils.simple_preprocess(str(sentence), deacc=True)) # deacc=True removes punctuations

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

def tfidf_prep(document):
    def make_bigrams(texts):
        return [bigram_mod[doc] for doc in texts]
    #prep stopwords
    more_stopwords = ("de, much, dont, im, liking, getting, gotten, ever, always, every, many, few, even, got, get, make, made, let, makes, im, say, said, better, best, really, would, something, came,much, oh, put, also, lyrics, lyric, comment, comments, lot, 2020, 2019, 2018, hey, hi, may, making, many, cause,com, youtube, etc, else, since, comes, come, www, la,que, re, el, use, used, still, much, ever, every, around,also, da, finally, sure, literal, literally, ooh, ya, every, album, let, -PRON-, pron, http, https").split(",")
    stops = list(stopwords.words('english'))
    stops.extend(more_stopwords)
    stops = [x.strip(' ') for x in stops]

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


def vectorize(df):
    print("Running TF-IDF on all data sets...")
    #prep stopwords
    more_stopwords = ("de, much, dont, im, liking, getting, gotten, ever, always, every, many, few, even, got, get, make, made, let, makes, im, say, said, better, best, really, would, something, came,much, oh, put, also, lyrics, lyric, comment, comments, lot, 2020, 2019, 2018, hey, hi, may, making, many, cause,com, youtube, etc, else, since, comes, come, www, la,que, re, el, use, used, still, much, ever, every, around,also, da, finally, sure, literal, literally, ooh, ya, every, album, let, -PRON-, pron, http, https").split(",")
    stops = list(stopwords.words('english'))
    stops.extend(more_stopwords)
    stops = [x.strip(' ') for x in stops]
    vectorizer = TfidfVectorizer(stop_words = stops, lowercase = True)#, tokenizer=LemmaTokenizer())
    vectors = vectorizer.fit_transform([df])
    feature_names = vectorizer.get_feature_names()
    dense = vectors.todense()
    denselist = dense.tolist()
    df = pd.DataFrame(denselist, columns=feature_names)
    #Clean results
    df = df.T
    df = df.rename(columns = {0: 'word_word'}, inplace = False)
    return df

def visualize(df):
    cloud_color = "summer"     #summer # Set wordcloud color scheme ('Wistia' or 'OrRd' are nice! YlGn is great for ayahuasca!)
    cloud_bg_color = "black"    # Set background color ("#182a2a" is a nice, dark green; #2b2d2f coal grey!)
    cloud_scale = 0             # Scale word size mainly by rank (0) to frequency (1) 
    cloud_horizontal = 1        # Ratio of word angles from horizontal (1) to vertical (0) 
    keys = tuple(df.index.tolist())
    values = tuple(df['word_word'])
    my_dict = dict(zip(keys,values))

    wordcloud = WordCloud(regexp=None, relative_scaling=cloud_scale, prefer_horizontal=cloud_horizontal, width = 500 , height = 500, background_color=cloud_bg_color, max_words=500, contour_width=0, colormap = cloud_color)

    wordcloud.generate_from_frequencies(my_dict)

    return wordcloud
