# -*- coding: utf-8 -*-
"""
Created on Wed Nov  7 11:33:27 2018

@author: Christian
"""

#%% Import packages 
import pandas as pd 
import numpy as np
from sentiment_module import sentiment
import nltk 
nltk.download('punkt')
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from nltk.tokenize import regexp_tokenize, TweetTokenizer
from sklearn import metrics
from sklearn.naive_bayes import MultinomialNB
import spacy 
import re, string
from sklearn.preprocessing import LabelEncoder
from textblob import TextBlob
import gensim
import matplotlib.pyplot as plt
#%% Read in data
tweets = pd.read_excel(r"C:\Users\Christian\Documents\Mario Party Project\mpt.xlsx",
                       sheet_name = "Sheet1")

#%% Remove retweets
tweets = tweets[tweets['retweet'] == 0]

#%% Focus in on the tweets themselves
tweets = pd.DataFrame(tweets["body"])

#%% Create function that cleans tweets (gets rid of #, links, and @)
def clean_tweet(tweet): 
   return str(' '.join(re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)", " ", tweet).split()))

#%% Clean the tweets
clean_tweets = tweets['body'].apply(np.vectorize(clean_tweet))

#%% Create function that makes all of the characters lowercase in a string
def lowercase(tweet):
    return (tweet.lower())

#%% Transform the series into a dataframe
clean_tweets = pd.DataFrame(clean_tweets)

#%% Apply the lowercase function
clean_tweets['body'] = clean_tweets['body'].apply(np.vectorize(lowercase))

#%%
'''Creating a Bag of Words'''
tknzr = TweetTokenizer()
all_tokens = [tknzr.tokenize(t) for t in tweets['body']]

# Make all of the strings lowercase
all_tokens = [t.lower() for list in all_tokens for t in list]

# Keep only strings that contain alphabetic characters
all_tokens = [t for t in all_tokens if t.isalpha()]

# Remove stopwords
from nltk.corpus import stopwords
all_tokens = [t for t in all_tokens if t not in stopwords.words('english')]

# Lemmatize the remaining strings
from nltk.stem import WordNetLemmatizer
wnl = WordNetLemmatizer()
all_tokens = [wnl.lemmatize(t) for t in all_tokens]

# Create a bag of words using the Counter() function, and look at the top 10 most common words
from collections import Counter
bow = Counter(all_tokens)
bow.most_common(10)
#%% Utilizing a function that classifies the polarity of a tweet
def analize_sentiment(tweet):
    '''
    Utility function to classify the polarity of a tweet
    using textblob.
    '''
    analysis = TextBlob(clean_tweet(tweet))
    if analysis.sentiment.polarity > 0:
        return 1
    elif analysis.sentiment.polarity == 0:
        return 0
    else:
        return -1
#%%
df = clean_tweets
df['SA'] = np.array([ analize_sentiment(tweet) for tweet in df['body'] ])
#%% Creating sub-dataframes for positive, neutral, and negative tweets
positive = df[df['SA'] == 1]
neutral = df[df['SA'] == 0]
negative = df[df['SA'] == -1]
#%%
print("Percentage of positive tweets: {}%".format(round(len(positive)*100/len(df['body']),2)))
print("Percentage of neutral tweets: {}%".format(round(len(neutral)*100/len(df['body']),2)))
print("Percentage de negative tweets: {}%".format(round(len(negative)*100/len(df['body']),2)))


#%% Selecting just tweeks that contain the word 'online'
online = clean_tweets
online = online[online['body'].str.contains("online", na=False)]

#%%
online['SA'] = np.array([ analize_sentiment(tweet) for tweet in online['body'] ])
#%%
pos_online = online[online['SA'] == 1]
neu_online = online[online['SA'] == 0]
neg_online = online[online['SA'] == -1]
#%%
print("Percentage of positive tweets about ONLINE: {}%".format(round(len(pos_online)*100/len(online['body']),2)))
print("Percentage of neutral tweets about ONLINE: {}%".format(round(len(neu_online)*100/len(online['body']),2)))
print("Percentage de negative tweets about ONLINE: {}%".format(round(len(neg_online)*100/len(online['body']),2)))
#%% Tweets without 'online'
offline = clean_tweets
offline = offline[offline['body'].str.contains("online") == False]
#%%
offline['SA'] = np.array([ analize_sentiment(tweet) for tweet in offline['body'] ])
#%%
pos_offline = offline[offline['SA'] == 1]
neu_offline = offline[offline['SA'] == 0]
neg_offline = offline[offline['SA'] == -1]
#%%
print("Percentage of positive tweets about ONLINE: {}%".format(round(len(pos_offline)*100/len(offline['body']),2)))
print("Percentage of neutral tweets about ONLINE: {}%".format(round(len(neu_offline)*100/len(offline['body']),2)))
print("Percentage de negative tweets about ONLINE: {}%".format(round(len(neg_offline)*100/len(offline['body']),2)))

#%% Create lists of entire tweets and of individual tokens of those tweets 
doc = [ ]
for i in range(0,len(clean_tweets['body'])):
    doc.append(clean_tweets.iloc[i][0])

#punc = re.compile( '[%s]' % re.escape( string.punctuation ) )
term_vec = [ ]

for d in doc:
    d = d.lower()
    
    term_vec.append( nltk.word_tokenize( d ) )
#%% Remove stop words from term vectors

stop_words = nltk.corpus.stopwords.words( 'english' )

for i in range( 0, len( term_vec ) ):
    term_list = [ ]

    for term in term_vec[ i ]:
        if term not in stop_words:
            term_list.append( term )

    term_vec[ i ] = term_list

# Print term vectors with stop words removed

#for vec in term_vec:
    #print( vec)
#%%# Porter stem remaining terms
porter = nltk.stem.porter.PorterStemmer()
stemmer = SnowballStemmer("english")
for i in range( 0, len( term_vec ) ):
    for j in range( 0, len( term_vec[ i ] ) ):
        term_vec[ i ][ j ] = porter.stem( term_vec[ i ][ j ] )

# Print term vectors with stop words removed

#for vec in term_vec:
    #print (vec)
#%%
stemmer = SnowballStemmer('english')
for i in range( 0, len( term_vec ) ):
    term_vec[i] = [stemmer.stem(t) for t in term_vec[i]]
#%% Retain only strings that contain alphabetical characters
for i in range( 0, len( term_vec ) ):
    term_vec[i] = [t for t in term_vec[i] if t.isalpha()]
    
        
#%%
#  Convert term vectors into gensim dictionary

dict = gensim.corpora.Dictionary( term_vec )

corp = [ ]
for i in range( 0, len( term_vec ) ):
    corp.append( dict.doc2bow( term_vec[ i ] ) )

#  Create TFIDF vectors based on term vectors bag-of-word corpora

tfidf_model = gensim.models.TfidfModel( corp )

tfidf = [ ]
for i in range( 0, len( corp ) ):
    tfidf.append( tfidf_model[ corp[ i ] ] )

#  Create pairwise document similarity index

n = len( dict )
index = gensim.similarities.SparseMatrixSimilarity( tfidf_model[ corp ], num_features = n )

#  Print TFIDF vectors and pairwise similarity per document

for i in range( 0, len( tfidf ) ):
    s = 'Doc ' + str( i + 1 ) + ' TFIDF:'

    for j in range( 0, len( tfidf[ i ] ) ):
        s = s + ' (' + dict.get( tfidf[ i ][ j ][ 0 ] ) + ','
        s = s + ( '%.3f' % tfidf[ i ][ j ][ 1 ] ) + ')'

    print(s)

for i in range( 0, len( corp ) ):
    print 'Doc', ( i + 1 ), 'sim: [ ',

    sim = index[ tfidf_model[ corp[ i ] ] ]
    for j in range( 0, len( sim ) ):
        print '%.3f ' % sim[ j ],

    print(']')
#%% Sentiment analysis using Dr. Healey's library
find_sent = term_vec

df_sent = pd.DataFrame(columns = ["Tweet","Valence","Arousal"])

for i in range(0, len(find_sent)):
    row_dict = {}
    tweet_sentiment = sentiment.sentiment(find_sent[i])
    row_dict["Tweet"] = clean_tweets.iloc[i][0]
    row_dict["Valence"] = tweet_sentiment["valence"]
    row_dict["Arousal"] = tweet_sentiment["arousal"]
    df_sent = df_sent.append(row_dict, ignore_index = True)
    
#%%
df_sent = df_sent[df_sent["Arousal"] != 0]
#%%
on_sent = df_sent[df_sent['Tweet'].str.contains('online')]
off_sent = df_sent[~df_sent['Tweet'].str.contains('online')]
#%% Scatterplot of the dispersion of Arousal and Valence for the tweets
plt.clf()
plt.scatter(x=off_sent["Arousal"], y=off_sent["Valence"],c="blue", label="General")
plt.scatter(x=on_sent["Arousal"], y=on_sent["Valence"],c="orange", label="Online")
plt.title("Valence and Arousal of Overall Tweets vs. 'Online' Tweets")
plt.xlabel("Arousal")
plt.ylabel("Valence")
plt.legend(loc='lower right')
plt.show()
#%%
plt.clf()

plt.subplot(2,2,1)
plt.hist(off_sent['Arousal'], bins=35, ec='black', color='purple')
plt.title('Arousal of General Tweets', fontsize=28)
plt.tick_params(labelsize=16)
plt.ylabel('Frequency', fontsize='16')
plt.text(5.5,2500, "Mean Arousal = 4.96", fontsize=20)

plt.subplot(2,2,2)
plt.hist(off_sent['Valence'], bins=35, ec='black', color='green')
plt.title('Valence of General Tweets', fontsize=28)
plt.tick_params(labelsize=16)
plt.ylabel('Frequency', fontsize='16')
plt.text(2,2400, "Mean Valence = 6.52", fontsize=20)

plt.subplot(2,2,3)
plt.hist(on_sent['Arousal'], bins=35, ec='black', color='purple')
plt.title('Arousal of Online Tweets', fontsize=28)
plt.tick_params(labelsize=16)
plt.ylabel('Frequency', fontsize='16')
plt.text(5.2,90, "Mean Arousal = 4.83", fontsize=20)

plt.subplot(2,2,4)
plt.hist(on_sent['Valence'], bins=35, ec='black', color='green')
plt.title('Valence of Online Tweets', fontsize=28)
plt.tick_params(labelsize=16)
plt.ylabel('Frequency', fontsize='16')
plt.text(3,90, "Mean Valence = 6.32", fontsize=20)



plt.show()
#%%
inner = pd.merge(df_sent,df_sentiment, on=['Tweet'])
#%%
off_ar = off_sent['Arousal'].mean()
on_ar = on_sent['Arousal'].mean()
off_val = off_sent['Valence'].mean()
on_val = on_sent['Valence'].mean()
#%%
#%%
luigi = df_sent[df_sent['Tweet'].str.contains('luigi')]
luigi = luigi[~luigi['Tweet'].str.contains('waluigi')]
peach = df_sent[df_sent['Tweet'].str.contains('peach')]
bowser = df_sent[df_sent['Tweet'].str.contains('bowser')]
goomba = df_sent[df_sent['Tweet'].str.contains('goomba')]
wario = df_sent[df_sent['Tweet'].str.contains('wario')]
waluigi = df_sent[df_sent['Tweet'].str.contains('waluigi')]
koopa = df_sent[df_sent['Tweet'].str.contains('koopa')]
yoshi = df_sent[df_sent['Tweet'].str.contains('yoshi')]
toad = df_sent[df_sent['Tweet'].str.contains('toad')]
rosalina = df_sent[df_sent['Tweet'].str.contains('rosalina')]
boo = df_sent[df_sent['Tweet'].str.contains('boo')]
c_list = [luigi,peach,bowser,goomba,wario,waluigi,koopa,yoshi,toad,rosalina,boo]
c_names = ['luigi','peach','bowser','goomba','wario','waluigi','koopa','yoshi','toad','rosalina','boo']

#%%
characters = pd.DataFrame(columns=['Name','Arousal','Valence'])

for i in range(0, len(c_list)):
    df_row = {}
    df_row['Name'] = c_names[i]
    df_row['Arousal'] = c_list[i]['Arousal'].mean()
    df_row['Valence'] = c_list[i]['Valence'].mean()
    characters = characters.append(df_row, ignore_index=True)
#%%
import seaborn as sns
# basic plot
p1=sns.regplot(data=characters, x="Arousal", y="Valence", fit_reg=False, marker="o", color="skyblue", scatter_kws={'s':400})
 
# add annotations one by one with a loop
for line in range(0,characters.shape[0]):
     p1.text(characters.Arousal[line]+0.01, characters.Valence[line], characters.Name[line], horizontalalignment='left', size='large', color='black', weight='semibold')
 
# see it
#sns.plt.show()








