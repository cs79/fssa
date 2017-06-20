# Federal Reserve Speeches Sentiment Analysis

import requests, nltk, re, string, nltk.data
from bs4 import BeautifulSoup
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import pandas as pd

#speechurl = 'https://www.federalreserve.gov/feeds/speeches.xml'
#pre2010 = 'https://www.federalreserve.gov/newsevents/speech/2010speech.htm'
pre2010dates = [str(date) for date in range(1996, 2011)]
post2010url = 'https://www.federalreserve.gov/newsevents/speeches.htm'
urlstem = 'https://www.federalreserve.gov'
tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
stemmer = PorterStemmer()

# scrape page, get speech links, change out link path for the pdf of the speech (if easiest)
# or look at html for a speech page and get the body part
# eliminate the references / extraneous information
# tokenize & clean
# score sentiment for each token (sentence)
# aggregate scores
# tag to speech ID with timestamp

# fetch pre-2010 speech page data
pre2010content = {}
for date in pre2010dates:
    # loop through url with date appended, then parse links and extract content
    dateurl = 'https://www.federalreserve.gov/newsevents/speech/{}speech.htm'.format(date)
    page = requests.get(dateurl)
    pre2010content[date] = page.content
    print('fetched content for {}'.format(date))

pre2010links = {}
# get content links for pre-2010 speech data
for date in pre2010content.keys():
    soup = BeautifulSoup(pre2010content[date], 'lxml')
    # ul id='speechIndex' is what we want
    # first li child has the date we want
    # first a has the link to the speech
    speechIndices = soup.find_all(name = 'li')[5:]
    # warning - may be fragile across years
    for li in speechIndices:
        # get the speech date
        tagtext = li.get_text()
        speechdate = re.search('[A-Z][a-z]+ \d+, \d+', tagtext)
        # get the link to the speech
        postfix = li.a.attrs['href']
        pre2010links[speechdate.group(0)] = urlstem + postfix
        print('Recorded url for speech "{}"'.format(li.a.get_text()))
    print('Finished parsing links for {}'.format(date))

# parse content links for pre-2010 speech data
pre2010speeches = {}
for k in pre2010links.keys():
    page = requests.get(pre2010links[k])
    soup = BeautifulSoup(page.content, 'html.parser')
    # "dirty" text, split into sentences
    speech = tokenizer.tokenize(soup.body.get_text())
    # remove footnotes (bootleg)
    i = 0
    while i < len(speech):
        if speech[i].startswith('Footnotes'):
            break
        else:
            i += 1
    cleaned = speech[:i]
    # clean the text
    cleaned = [re.sub('\'|-', ' ', sent) for sent in cleaned]
    punct = re.compile('[%s]' % re.escape(string.punctuation))
    cleaned = [re.sub(punct, '', sent) for sent in cleaned]
    cleaned = [re.sub('\s+', ' ', sent) for sent in cleaned]
    cleaned = [re.sub('\d', '', sent) for sent in cleaned]
    # lowercase everything
    cleaned = [sent.lower() for sent in cleaned]
    # remove stopwords
    for i in range(len(cleaned)):
        sent = cleaned[i]
        clean_sent = []
        for word in sent.split():
            if word not in stopwords.words('english'):
                clean_sent.append(word)
        clean_sent = ' '.join(clean_sent)
        cleaned[i] = clean_sent
    # stem words
    cleaned = [' '.join([stemmer.stem(word) for word in sent.split()]) for sent in cleaned]
    # index list of cleaned speech text tokens into dict
    pre2010speeches[k] = cleaned
    print('Cleaned speech data for {}'.format(k))

# make cleaned data portable
pre2010data = pd.DataFrame(columns = ['Date', 'Token'])
i = 0
for date in pre2010speeches.keys():
    for j in range(len(pre2010speeches[date])):
        pre2010data.loc[i, 'Date'] = date
        pre2010data.loc[i, 'Token'] = pre2010speeches[date][j]
        i += 1
pre2010data.sort('Date')
# convert to timestamp at some point
pre2010data.index = range(len(pre2010data))
pre2010data.to_csv('C:/Users/Alex/Dropbox/Projects/fssa/pre2010data.csv')

# need to do a similar analysis of tags to look for the part containing the speech body
# also may need to use re to strip out things like reference section, etc.
# once we have the body, clean it (not necessarily in correct order):
    # tokenize by sentences
    # strip excess whitespace
    # lowercase
    # remove stopwords
    # stem words
# once we have lists of sentences per speech, use vader to get sentiment
# average across sentences (maybe?) for each day -- think about this more
# pack into dataframe with date index and a column for sentiment score for that day's speech

# then do this for post-2010 speeches - trickier html scheme for this


# once we have timeseries sentiment:
# series will be very sparse - need to fill forward to make continuous for regression
# options:
    # flat forward-fill (step function sentiment)
    # flat ffill with some decay function added
    # decay functions:
        # long-term mean reversion (previous n months/years)
        # logarithmic decay
        # linear decay
        # relu-style decay (jointed spline)
    # point-to-point connected (DO NOT DO THIS)

# could be interesting to compare effects of speech sentiment vs. vix on, say, SPX
# for each inter-speech period, could compute a weighted combination of standardized
# sentiment level and VIX level for the same backward-looking base period (say, 1 year)
# then test each of these (and / or combinations thereof) vs. SPX
    # sentiment index
    # VIX level
    # weighted combo
