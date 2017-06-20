# Federal Reserve Speeches Sentiment Analysis

import requests, nltk, re
from bs4 import BeautifulSoup
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import pandas as pd

#speechurl = 'https://www.federalreserve.gov/feeds/speeches.xml'
#pre2010 = 'https://www.federalreserve.gov/newsevents/speech/2010speech.htm'
pre2010dates = [str(date) for date in range(1996, 2011)]
post2010url = 'https://www.federalreserve.gov/newsevents/speeches.htm'
urlstem = 'https://www.federalreserve.gov'

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
