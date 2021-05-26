# -*- coding: utf-8 -*-
"""
Created on Wed May 26 01:03:20 2021

@author: Admin
"""
import streamlit as st
import datetime as dt
import pickle
import nltk
nltk.download('stopwords')
from nltk.classify import SklearnClassifier
#from twitterscraper import query_tweets
from wordcloud import WordCloud,STOPWORDS

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


st.subheader('Single tweet classification')

# Get sentence input, preprocess it, and convert to flair.data.Sentence format
tweet_input = st.text_input('Tweet:')

if tweet_input != '':
    
    # Make predictions
    with st.spinner('Predicting...'):
        res=classifier.classify(extract_features(tweet_input.split()))
  
    st.write('Prediction:', res)
   