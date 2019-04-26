# showing visualization
# visualization.py
import csv
import numpy as np  
import pandas as pd  
import re  
import nltk  
import matplotlib.pyplot as plt 
# for wordcloud
from wordcloud import WordCloud, STOPWORDS 

from scipy.misc import imread
from textblob import TextBlob


# read data from csv file
data = pd.read_csv('analysis21.csv')
'''
data.Sentiment.value_counts().plot(kind='pie', autopct='%1.0f%%', colors=["red", "yellow", "green"])
plt.show()
sentiment = data.groupby(['Subjectiivity','Sentiment']).Sentiment.count().unstack()  
sentiment.plot(kind='bar')
plt.show()
import seaborn as sns

sns.barplot(x='Sentiment', y='Subjectiivity' , data=data)

plt.show()
'''

val1=(data.loc[data['Sentiment'] == 'Negative'])
'''
#v=(data.loc[data['SVM'] == 'negative'])
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
stop_words = set(stopwords.words('english'))
wordfreq={}
for tweet in val1['Tweet']:
    word_tokens = word_tokenize(tweet)
    filtered_sentence = [w for w in word_tokens if not w in stop_words]
    filtered_sentence = []
    for w in word_tokens:
        if w not in stop_words:
            if(w not in wordfreq):
                wordfreq[w]=0
            wordfreq[w]+=1
            
neg_words = list(wordfreq.keys())
print('-'*60)
print("NEGATIVE WORDS")
for i in neg_words:
    print(i)
    
val2=(data.loc[data['Sentiment'] == 'Positive'])
#v2=(data.loc[data['SVM'] == 'positive'])
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
stop_words = set(stopwords.words('english'))
positivewordfreq={}
for tweet in val2['Tweet']:
    word_tokens = word_tokenize(tweet)
    filtered_sentence = [w for w in word_tokens if not w in stop_words]
    filtered_sentence = []
    for w in word_tokens:
        if w not in stop_words:
            if(w not in positivewordfreq):
                positivewordfreq[w]=0
            positivewordfreq[w]+=1
pos_words = list(positivewordfreq.keys())
print('-'*60)
print("POSITIVE WORDS")
for i in pos_words:
    print(i)            
val3=(data.loc[data['Sentiment'] == 'Neutral'])
#v3=(data.loc[data['SVM'] == 'neutral'])
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
stop_words = set(stopwords.words('english'))
neutralwordfreq={}
for tweet in val1['Tweet']:
    word_tokens = word_tokenize(tweet)
    filtered_sentence = [w for w in word_tokens if not w in stop_words]
    filtered_sentence = []
    for w in word_tokens:
        if w not in stop_words:
            if(w not in neutralwordfreq):
                neutralwordfreq[w]=0
            neutralwordfreq[w]+=1
            
neu_words = list(neutralwordfreq.keys())
print('-'*60)
print("NEUTRAL WORDS")
for i in neu_words:
    print(i)
    '''
comment_words = ' '
stopwords = set(STOPWORDS) 
negativeTweets =[]
for i in val1['Tweet']:
    negativeTweets.append(i)


for i in negativeTweets:
    tokens = i.split

for j in tokens:
    comment_words = comment_words +j + ' '
wordcloud = WordCloud(width = 800, height = 800, 
                background_color ='white', 
                stopwords = stopwords, 
                min_font_size = 10).generate(comment_words) 
  
# plot the WordCloud image                        
plt.figure(figsize = (8, 8), facecolor = None) 
plt.imshow(wordcloud) 
plt.axis("off") 
plt.tight_layout(pad = 0) 
  
plt.show() 







    
'''
    
logomask = imread('twitter_mask.png')
print(val1)
negativeTweets =[]
for i in val1['Tweet']:
    negativeTweets.append(i)
    
for j in negativeTweets:
    print(j)

    
wordcloud = WordCloud(stopwords=STOPWORDS,background_color='white',mask = logomask,max_words=250).generate(negativeTweets)  
  
plt.imshow(wordcloud.recolor(color_func=grey_color_func, random_state=3))
plt.axis('off')
plt.savefig('./tweetcloud2.png', dpi=300)
plt.show()  
'''
print('Done')
