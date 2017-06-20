import pandas as pd
import requests, nltk, re
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import sklearn.svm
import nltk.classify.util
from nltk.classify import NaiveBayesClassifier
from nltk.corpus import names

with open('FOMCpresconf20170315.txt', 'r') as myfile:
    pc20170315 = myfile.read().replace('\n', '')

with open('YellenSpeech20170315.txt', 'r') as myfile:
    speech20170315 = myfile.read().replace('\n', '')

spx_hist = pd.read_csv('prices.csv', header = 0, index_col = 0)

pc_words = pc20170315.split(' ')

def get_word_features(wordlist):
    wordlist = nltk.FreqDist(wordlist)
    word_features = wordlist.keys()
    return(word_features)

def word_feats(words):
    return(dict([(word, True) for word in words]))

positive_vocab = ['awesome', 'outstanding', 'fantastic', 'terrific', 'good', \
                    'nice', 'great', ':)']
negative_vocab = ['bad', 'terrible', 'useless', 'hate', ':(']
neutral_vocab = ['movie', 'the', 'sound', 'was', 'is', 'actors', 'did', 'know' \
                    'words', 'not']

positive_features = [(word_feats(pos), 'pos') for pos in positive_vocab]
negative_features = [(word_feats(neg), 'neg') for neg in negative_vocab]
neutral_features = [(word_feats(neu), 'neu') for neu in neutral_vocab]

train_set = negative_features + positive_features + neutral_features

classifier = NaiveBayesClassifier.train(train_set)

# Predict
neg = 0
pos = 0

for word in pc_words:
    classResult = classifier.classify(word_feats(word))
    if classResult == 'neg':
        neg += 1
    if classResult == 'pos':
        pos += 1

print('Positive: {}'.format(str(pos / len(pc_words))))
print('Negative: {}'.format(str(neg / len(pc_words))))

pc_sentences = pc20170315.split('.')
sid = SentimentIntensityAnalyzer
for sentence in pc_sentences:
    print(sentence)
    ss = sid.polarity_scores(sentence)
    for k in sorted(ss):
        print('{0}: {1}, '.format(k, ss[k]), end = '')
