# -*- coding: utf-8 -*-
"""
Created on Wed May 26 01:03:20 2021

@author: Admin
"""
import streamlit as st
import datetime as dt
import pickle
import nltk
import os
import re
import lxml
import asyncio
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from nltk.tokenize import WordPunctTokenizer
from bs4 import BeautifulSoup
import pandas as pd
import twint

from nltk.corpus import stopwords
import nest_asyncio
nltk.download('stopwords')
from nltk.classify import SklearnClassifier

from wordcloud import WordCloud,STOPWORDS
st.set_option('deprecation.showPyplotGlobalUse', False)


def showWordCloud (df,sentiment, stopwords):
  tweets = df[df.sentiment == sentiment]
  string = []
  for t in tweets.tweet:
      string.append(t)
  string = pd.Series(string).str.cat(sep=' ')

  wordcloud = WordCloud(stopwords=stopwords, background_color="white").generate(string)
  plt.figure()
  plt.imshow(wordcloud, interpolation="bilinear")
  plt.axis("off")
  st.pyplot()
  
with open('tweets.pkl', 'rb') as f:
    tweets = pickle.load(f)

# Extracting word features
def get_words_in_tweets(tweets):
    all = []
    for (words, sentiment) in tweets:
        all.extend(words)
    return all
 
def get_word_features(wordlist):
    wordlist = nltk.FreqDist(wordlist)
    features = wordlist.keys()
    return features
w_features = get_word_features(get_words_in_tweets(tweets))
 
def extract_features(document):
    document_words = set(document)
    features = {}
    for word in w_features:
        features['contains(%s)' % word] = (word in document_words)
    return features


st.title('Twitter and Facebook Sentiment Analysis')

with st.spinner('Loading classification model...'):
    filename='naive_finalized_model.sav'
    classifier = pickle.load(open(filename, 'rb'))
#print("Classifier accuracy percent:",(nltk.classify.accuracy(classifier, testing_set))*100) 


st.subheader('Single tweet/status classification')

# Get sentence input, preprocess it, and convert to flair.data.Sentence format
tweet_input = st.text_input('Tweet:')

if tweet_input != '':
    
    # Make predictions
    with st.spinner('Predicting...'):
        res=classifier.classify(extract_features(tweet_input.split()))
  
    st.write('Prediction:', res)
    

st.subheader('Search Twitter for Query')
def load_tweet(keyword, limit):
    #nest_asyncio.apply()
    t=twint.Config()
    t.Search=query
    t.Store_csv=True
    t.Limit=limit
    t.lang="en"
    t.Geo = "40.75543016826262, -73.84925768925918,100km"
    t.Custom_csv=['id','user_id','username','text']
    t.Output=('tweet_data.csv')
    
    asyncio.set_event_loop(asyncio.new_event_loop())
    twint.run.Search(t)
    
    tweet_data=pd.read_csv('tweet_data.csv')
    tweet_data['sentiment']=' '
    tweet_data=tweet_data[['tweet','sentiment']]
    
    os.remove('tweet_data.csv')
    return tweet_data

def tweet_cleaner(text):
  tok = WordPunctTokenizer()
  pat1 = r'@[A-Za-z0-9]+'
  pat2 = r'https?://[A-Za-z0-9./]+'
  pat3 = r'pic.twitter.com/[A-Za-z0-9./]+'
  combined_pat = r'|'.join((pat1, pat2, pat3))
  soup = BeautifulSoup(text, 'lxml')
  souped = soup.get_text()
  stripped = re.sub(combined_pat, '', souped)
  try:
    clean = stripped.decode("utf-8-sig").replace(u"\ufffd", "?")
  except:
    clean = stripped
  letters_only = re.sub("[^a-zA-Z]", " ", clean)
  lower_case = letters_only.lower()
  words = tok.tokenize(lower_case)
  return (" ".join(words)).strip()





# Get user input
query = st.text_input('Enter a word to scrape and analyze sentiment ', '#')
limit = st.number_input('Input Limit for Screep', 100, step=50)
if st.button('Analyze'):
    if query != "" and limit != "":
        st.subheader("Loading tweets...")
        tweet_data=load_tweet(query,limit)
        st.write(tweet_data.head(10))
        predictions_count = {'Positive': 0, 'Negative': 0,'Neutral':0}

        
    if query != "" and limit != "":
        st.subheader("Cleaningtweets...")
        removing_data = st.text('Removing not required characters...')
        tweet_data['tweet'] = tweet_data['tweet'].apply(lambda text: tweet_cleaner(text))
        removing_data.text('Removing... done!')
        st.table(tweet_data['tweet'].head(10))



        
    if query != "" and limit != "":
        st.subheader("Predicting Sentiments...")
        for index,row in tweet_data.iterrows():
            
            res=classifier.classify(extract_features(row.tweet.split()))
            predictions_count[res]+=1
            row.sentiment=res
        st.write('Analyzed data:')
        st.write(tweet_data)
        print(predictions_count)
    
        count=predictions_count.values()
        st.write('Positive :', predictions_count['Positive']/sum(count))
        st.write('Negative:', predictions_count['Negative']/sum(count))
        st.write('Neutral :', predictions_count['Neutral']/sum(count))

    if query != "" and limit != "":
        st.subheader("Sentiment Analysis Percentage")
        labels = 'Positive', 'Netral', 'Negative'
        pos = tweet_data[tweet_data.sentiment == 'Positive'].shape[0]
        net = tweet_data[tweet_data.sentiment == 'Neutral'].shape[0]
        neg = tweet_data[tweet_data.sentiment == 'Negative'].shape[0]
        sizes = [pos, net, neg]
        colors = ['#ff9999', 'gold', '#ff9999']
        explode = (0.1, 0, 0)  # explode 1st slice
        fig1, ax1 = plt.subplots()
        ax1.pie(sizes, explode=explode, labels=labels, autopct='%1.1f%%',shadow=True, startangle=90)
        ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
        st.pyplot()

    if query != "" and limit != "":
        stopword = set(stopwords.words("english"))
        st.subheader("Word Cloud Positive Tweet")
        asyncio.set_event_loop(asyncio.new_event_loop())
        showWordCloud(tweet_data, 'Positive', stopword)

        st.subheader("Word Cloud Neutral Tweet")
        
        asyncio.set_event_loop(asyncio.new_event_loop())
        showWordCloud(tweet_data, 'Neutral', stopword)

        st.subheader("Word Cloud Negative Tweet")
        asyncio.set_event_loop(asyncio.new_event_loop())
        showWordCloud(tweet_data, 'Negative', stopword)
        
    
