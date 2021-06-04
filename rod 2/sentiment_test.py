import pandas as pd
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk
nltk.download('vader_lexicon')
import numpy as np

sid = SentimentIntensityAnalyzer()
data = pd.read_csv("test.csv")

data['scores'] = data['Comment'].apply(lambda comment: sid.polarity_scores(comment))
data['compound'] = data['scores'].apply(lambda x: x.get('compound'))

if st.button('Sentiment run'):
        import pandas as pd
        from nltk.sentiment.vader import SentimentIntensityAnalyzer
        import nltk
        nltk.download('vader_lexicon')
        import numpy as np
        sid = SentimentIntensityAnalyzer()
        if hasattr(ss, 'data1'):
            ss.data1['scores'] = ss.data1['Comment'].apply(lambda comment: sid.polarity_scores(comment))
            ss.data1['compound'] = ss.data1['scores'].apply(lambda x: x.get('compound'))
        if hasattr(ss, 'data2'):
            ss.data2['scores'] = ss.data2['Comment'].apply(lambda comment: sid.polarity_scores(comment))
            ss.data2['compound'] = ss.data2['scores'].apply(lambda x: x.get('compound'))
        if hasattr(ss, 'data3'):
            ss.data3['scores'] = ss.data3['Comment'].apply(lambda comment: sid.polarity_scores(comment))
            ss.data3['compound'] = ss.data3['scores'].apply(lambda x: x.get('compound'))

