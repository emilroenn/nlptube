import os 
import pickle
import pandas as pd
from pandas import DataFrame
#import google.oauth2.credentials
#from googleapiclient.discovery import build
#from googleapiclient.errors import HttpError
#from google_auth_oauthlib.flow import InstalledAppFlow
#from google.auth.transport.requests import Request
from youtube_transcript_api import YouTubeTranscriptApi
import matplotlib.pyplot as plt
import streamlit as st

from stqdm import stqdm
from wordcloud import WordCloud
from PIL import Image



###############################


#CLIENT_SECRETS_FILE = "client_secret.json"

#SCOPES = ['https://www.googleapis.com/auth/youtube.force-ssl']
#API_SERVICE_NAME = 'youtube'
#API_VERSION = 'v3'



def get_authenticated_service():
    credentials = None
    SCOPES = ['https://www.googleapis.com/auth/youtube.force-ssl']
    CLIENT_SECRETS_FILE = "client_secret.json"
    API_SERVICE_NAME = 'youtube'
    API_VERSION = 'v3'
    API_key = "AIzaSyA_OuQywrdv4MSgel1xFnnAvdgR0zvR58M"
    code = "4/1AfDhmrgsalD6LJlRuWliCzO0o7o8qLouK6Lg3plZMCW2H5dyeDAYUGrdfGY"

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

#@st.cache(allow_output_mutation=True,persist=True, suppress_st_warning=True)
def get_data(query, results):

    os.environ['OAUTHLIB_INSECURE_TRANSPORT'] = '1'
    service = get_authenticated_service()

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
        #global total_scraped
        #total_scraped = total_scraped + 1
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