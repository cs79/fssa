# Federal Reserve Speeches Sentiment Analysis

import requests, nltk, re, string, nltk.data
from bs4 import BeautifulSoup
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import pandas as pd
from textblob import TextBlob
import pickle

pre2010dates = [str(date) for date in range(1996, 2011)]
post2010dates = [str(date) for date in range(2011, 2018)]
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

# get content links and speakers for pre-2010 speech data
pre2010links = {}
pre2010speakers = {}
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
        speechdate = re.search('[A-Z][a-z]+ \d{1,2}, \d{4}', tagtext)
        # get the link to the speech
        postfix = li.a.attrs['href']
        pre2010links[speechdate.group(0)] = urlstem + postfix
        # get the speaker info
        speaker = li.find('div', attrs = {'class': 'speaker'}).get_text()
        speaker = re.sub('\s{2,}', '', speaker)
        # fix for weird error I'm seeing:
        speaker = re.sub('\xa0', ' ', speaker)
        pre2010speakers[speechdate.group(0)] = speaker
        print('Recorded url and speaker for speech "{}"'.format(li.a.get_text()))
    print('Finished parsing links and speakers for {}'.format(date))

# parse content links for pre-2010 speech data
pre2010speeches = {}
NOSTEM_pre2010 = {}
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
    NOSTEM_pre2010[k] = cleaned.copy()
    # stem words
    cleaned = [' '.join([stemmer.stem(word) for word in sent.split()]) for sent in cleaned]
    # index list of cleaned speech text tokens into dict
    pre2010speeches[k] = cleaned
    print('Cleaned speech data for {}'.format(k))

# serialize so that we don't have to pull this again if it crashes
with open('C:/Users/Alex/Dropbox/Projects/fssa/pre2010.pickle', 'wb') as f:
    pickle.dump(pre2010speeches, f, pickle.HIGHEST_PROTOCOL)
with open('C:/Users/Alex/Dropbox/Projects/fssa/pre2010_NOSTEM.pickle', 'wb') as f:
    pickle.dump(NOSTEM_pre2010, f, pickle.HIGHEST_PROTOCOL)

# make cleaned data portable
# pretty sure this is incredibly inefficient
pre2010data = pd.DataFrame(columns = ['Date', 'Token'])
i = 0
for date in pre2010speeches.keys():
    for j in range(len(pre2010speeches[date])):
        pre2010data.loc[i, 'Date'] = date
        pre2010data.loc[i, 'Token'] = pre2010speeches[date][j]
        i += 1
pre2010data['Datestamp'] = [pd.Timestamp(date) for date in pre2010data['Date']]
pre2010data.sort_values('Datestamp', inplace = True)
pre2010data.index = range(len(pre2010data))

# TODO: NOSTEM version of this file
pre2010data_NOSTEM = pd.DataFrame(columns = ['Date', 'Token'])
i = 0
for date in NOSTEM_pre2010.keys():
    for j in range(len(NOSTEM_pre2010[date])):
        pre2010data_NOSTEM.loc[i, 'Date'] = date
        pre2010data_NOSTEM.loc[i, 'Token'] = NOSTEM_pre2010[date][j]
        i += 1
    # debug
    print('Finished parsing data for {}'.format(date))
pre2010data_NOSTEM['Datestamp'] = [pd.Timestamp(date) for date in pre2010data_NOSTEM['Date']]
pre2010data_NOSTEM.sort_values('Datestamp', inplace = True)
pre2010data_NOSTEM.index = range(len(pre2010data_NOSTEM))
pre2010data_NOSTEM['Speaker'] = [pre2010speakers[date] for date in pre2010data_NOSTEM['Date']]

# TODO extract a couple speeches as examples to be hand-annotated
# can use pre2010speakers as a dataframe to look up speakers for the pre2010data_NOSTEM df
# then pull out a few days worth where the speaker was interesting, and email to Warren, Kevin, Harrison
samples = pre2010data_NOSTEM[pre2010data_NOSTEM.Speaker == 'Chairman Ben S. Bernanke']
sample1 = samples[samples['Date'] == 'May 15, 2007']
sample2 = samples[samples['Date'] == 'November 14, 2008']
sample1.append(sample2).to_csv('C:/Users/Alex/Dropbox/Projects/fssa/sample_bernanke_speeches.csv')

# post-2010 data
post2010content = {}
for date in post2010dates:
    # loop through url with date appended, then parse links and extract content
    dateurl = 'https://www.federalreserve.gov/newsevents/speech/{}-speeches.htm'.format(date)
    page = requests.get(dateurl)
    post2010content[date] = page.content
    print('fetched content for {}'.format(date))

# get content links and speakers for post-2010 speech data
post2010links = {}
post2010speakers = {}
for date in post2010content.keys():
    soup = BeautifulSoup(post2010content[date], 'html.parser')
    # need to find tag for date info and link to speech content
    # we need divs where the class ends with eventlist__event and eventlist__time
    timere = re.compile('.+eventlist__time')
    times = soup.find_all('div', attrs = {'class': timere})
    timelist = []
    eventre = re.compile('.+eventlist__event')
    events = soup.find_all('div', attrs = {'class': eventre})
    eventlinks = []
    for time in times:
        timelist.append(time.time.get_text())
    for event in events:
        eventlinks.append(event.a.get('href'))
    speakers = []
    for event in events:
        speakers.append(event.find('p', attrs = {'class': 'news__speaker'}).get_text())
    assert len(timelist) == len(eventlinks) == len(speakers)
    for i in range(len(timelist)):
        post2010links[timelist[i]] = urlstem + eventlinks[i]
        post2010speakers[timelist[i]] = speakers[i]
        print('Recorded url and speaker for speech "{}"'.format(events[i].a.get_text()))
    print('Finished parsing links for {}'.format(date))

# parse content links for post-2010 speech data
post2010speeches = {}
NOSTEM_post2010 = {}
for k in post2010links.keys():
    page = requests.get(post2010links[k])
    soup = BeautifulSoup(page.content, 'html.parser')
    # "dirty" text, split into sentences
    speech = tokenizer.tokenize(soup.body.get_text())
    # remove header and footer boilerplate (bootleg)
    cleaned = speech[4:-1]
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
    # elimintate a couple other things
    better = []
    for i in range(len(cleaned)):
        if cleaned[i].startswith('return text') or cleaned[i].startswith('see ') \
            or cleaned[i] == '' or (len(cleaned[i]) < 12) and cleaned[i].endswith('pp'):
            continue
        else:
            better.append(cleaned[i])
    NOSTEM_post2010[k] = better.copy()
    # stem words
    cleaned = [' '.join([stemmer.stem(word) for word in sent.split()]) for sent in better]
    # index list of cleaned speech text tokens into dict
    post2010speeches[k] = cleaned
    print('Cleaned speech data for {}'.format(k))

# make cleaned data portable
post2010data = pd.DataFrame(columns = ['Date', 'Token'])
i = 0
for date in post2010speeches.keys():
    for j in range(len(post2010speeches[date])):
        post2010data.loc[i, 'Date'] = date
        post2010data.loc[i, 'Token'] = post2010speeches[date][j]
        i += 1
post2010data['Datestamp'] = [pd.Timestamp(date) for date in post2010data['Date']]
post2010data.sort_values('Datestamp', inplace = True)
post2010data.index = range(len(pre2010data), len(pre2010data) + len(post2010data))

# TODO: NOSTEM version of this file
post2010data_NOSTEM = pd.DataFrame(columns = ['Date', 'Token'])
i = 0
for date in NOSTEM_post2010.keys():
    for j in range(len(NOSTEM_post2010[date])):
        post2010data_NOSTEM.loc[i, 'Date'] = date
        post2010data_NOSTEM.loc[i, 'Token'] = NOSTEM_post2010[date][j]
        i += 1
    # debug
    print('Finished parsing data for {}'.format(date))
post2010data_NOSTEM['Datestamp'] = [pd.Timestamp(date) for date in post2010data_NOSTEM['Date']]
post2010data_NOSTEM.sort_values('Datestamp', inplace = True)
post2010data_NOSTEM['Speaker'] = [post2010speakers[date] for date in post2010data_NOSTEM['Date']]
post2010data_NOSTEM.index = range(len(pre2010data_NOSTEM), len(pre2010data_NOSTEM) + len(post2010data_NOSTEM))

# combine data
speechdata = pre2010data.append(post2010data)
# TODO: NOSTEM version of this file
speechdata_NOSTEM = pre2010data_NOSTEM.append(post2010data_NOSTEM)

# get links dataframe
links = pd.DataFrame()
i = 0
for day in pre2010links.keys():
    links.loc[i, 'Date'] = str(pd.Timestamp(day).date())
    links.loc[i, 'URL'] = pre2010links[day]
    i += 1
for day in post2010links.keys():
    links.loc[i, 'Date'] = str(pd.Timestamp(day).date())
    links.loc[i, 'URL'] = post2010links[day]
    i += 1
links.sort_values('Date', inplace = True)
links.index = range(len(links))
links.to_csv('C:/Users/Alex/Dropbox/Projects/fssa/links.csv')

# get speakers dataframe
speakers = pd.DataFrame()
i = 0
for day in pre2010speakers.keys():
    speakers.loc[i, 'Date'] = str(pd.Timestamp(day).date())
    speakers.loc[i, 'Speaker'] = pre2010speakers[day]
    i += 1
for day in post2010speakers.keys():
    speakers.loc[i, 'Date'] = str(pd.Timestamp(day).date())
    speakers.loc[i, 'Speaker'] = post2010speakers[day]
    i += 1
speakers.sort_values('Date', inplace = True)
speakers.index = range(len(speakers))
speakers.to_csv('C:/Users/Alex/Dropbox/Projects/fssa/speakers.csv')

# get compound sentiment
vaderanalyzer = SentimentIntensityAnalyzer()
speechdata['Sentiment_Vader'] = [vaderanalyzer.polarity_scores(speechdata['Token'][i])['compound'] \
                            for i in range(len(speechdata))]

# compare with textblob sentiment
speechdata['Sentiment_TextBlob'] = [TextBlob(i).sentiment.polarity for i in speechdata['Token']]

# NOSTEM version
# get compound sentiment
vaderanalyzer = SentimentIntensityAnalyzer()
speechdata_NOSTEM['Sentiment_Vader'] = [vaderanalyzer.polarity_scores(speechdata_NOSTEM['Token'][i])['compound'] \
                            for i in range(len(speechdata_NOSTEM))]

# compare with textblob sentiment
speechdata_NOSTEM['Sentiment_TextBlob'] = [TextBlob(i).sentiment.polarity for i in speechdata_NOSTEM['Token']]

speechdata_NOSTEM.to_csv('C:/Users/Alex/Dropbox/Projects/fssa/speechdata_NOSTEM.csv')

# quick viz
import matplotlib.pyplot as plt
plt.hist(speechdata['Sentiment'], bins = 50)
plt.show()

len(speechdata[speechdata['Sentiment'] == 0])
len(speechdata)
# so approximately 35% of sentence tokens have a 0 sentiment index - maybe best to throw these away
signal = speechdata[speechdata['Sentiment'] != 0]
signal.index = range(len(signal))

# average by date for days where we are getting some signal (probably some better method we should use)
groupedts = signal.groupby('Datestamp').mean()
groupedts.index = pd.DatetimeIndex(groupedts.index)

# quick viz of forward-filled timeseries for available dates; no decay function
groupedts.asfreq('B').ffill().plot()
plt.show()
# alongside standardized S&P 500 historical adjusted closing prices / volumes
sp = pd.read_csv('SP500historical.csv', header = 0, index_col = 0)
sp = sp[['Adj Close', 'Volume']]
sp.rename(columns = lambda x: 'S&P 500 - ' + x, inplace = True)
sp['S&P 500 - Adj Close'] = sp['S&P 500 - Adj Close'] / sp['S&P 500 - Adj Close'].max()
sp['S&P 500 - Volume'] = sp['S&P 500 - Volume'] / sp['S&P 500 - Volume'].max()
sp = sp.join(groupedts, how = 'outer')
sp.ffill().plot()
plt.show()
# nothing immediately clear but we could do something like better transforms
# (e.g. GARCH for S&P prices), VAR Granger test, etc. and see if we find anything more interesting


# TODO: compare TextBlob sentiment vs. vader

# NOTES

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
