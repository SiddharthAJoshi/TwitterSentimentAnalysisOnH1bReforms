import pandas as pd
import GetOldTweets3 as got
from textblob import TextBlob
from textblob import Blobber
from textblob.sentiments import NaiveBayesAnalyzer
import tweepy
import csv
import re
import sys

# authentication
def authenticate():
    consumer_key = 'K3tU5xdiI34h7uojwATLtZ8a4'
    consumer_secret = 'LDJk25o0p0qybcjeSRsoXVpFN5f5HgD0E9fmxG3fScVCkCliPG'
    access_token = '1295410230-Yg6LJ61XzDckPLSE0YUGmdw5dtlZzOqRLQdmUez'
    access_token_secret = 'nxmlufp6mhTNB8OdcV57YSHHPxhKkeyVhbXIbsvth8cFH'
    try:
        auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
        auth.set_access_token(access_token, access_token_secret)
        api = tweepy.API(auth)

    except tweepy.TweepError as e:
        print("Error : " + str(e))

    query = 'H1B visa'
    since = '2016-04-25' # for trial only small number
    until = '2019-04-20'
    print("Start...")
    getOldTweets(query,since, until)


# extracting old tweets
def getOldTweets(query,since, until):
    print("inside getOldTweets")
    tweetCriteria = got.manager.TweetCriteria().setQuerySearch(query).setSince(since).setUntil(until)
    tweet = got.manager.TweetManager.getTweets(tweetCriteria)

    processTweets(query,tweet)

# before analysing for sentiments the tweets are processed and cleaned
def processTweets(query,tweet):
    print("inside processTweets")
    data=[]
    ''''''
    positive = 0
    negative = 0
    neutral = 0

    for t in tweet:
        text=t.text
        cleanedTweet= cleanTweets(text)
        
        analysis= TextBlob(cleanedTweet)
        a,b = analysis.sentiment
        print (analysis.sentiment)
        if(analysis.sentiment.polarity < 0):
            polarity = 'Negative'
            negative += 1
        elif(analysis.sentiment.polarity == 0):
            polarity = 'Neutral'
            neutral +=1
        elif(analysis.sentiment.polarity > 0):
            polarity = 'Positive'
            positive += 1
        print (polarity)

        dic={}
        
        dic['Sentiment']=polarity
        dic['Tweet']=cleanedTweet.lower()
        dic['Polarity'] =  a
        dic['Subjectiivity'] = b
        data.append(dic)
    df=pd.DataFrame(data)
    text = df["Tweet"]
    for i in range(0,len(text)):
        txt = ' '.join(word for word in text[i] .split() if not word.startswith('https:'))
        df.set_value(i, 'Tweet1', txt)
    df.drop_duplicates('Tweet1', inplace=True)
    df.reset_index(drop = True, inplace=True)
    df.drop('Tweet', axis = 1, inplace = True)
    df.rename(columns={'Tweet1': 'Tweet'}, inplace=True) 
    df.to_csv('analysis21.csv',index=None)

# cleaning tweets before classifying
def cleanTweets(tweet):
    return ' '.join(re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)|(RT)", " ", tweet).split())


if __name__ == "__main__":
    authenticate()
    
