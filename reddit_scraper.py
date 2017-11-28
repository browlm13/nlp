#!/usr/bin/env python

"""
Reddit Scraper

requirements: 
	-Newspaper3k, https://github.com/codelucas/newspaper, http://newspaper.readthedocs.io/en/latest/

Notes: 	
	newspaper offers nlp summary
	article.nlp()
	print(article.summary)

"""
#internal
import json, requests
import string
import re
import json
import logging

# mylib
from rake_sentence_ranking import *
#import normalizers as norm

# external
from newspaper import Article


#
# 								Settings
#

path = 'data_set/'
subreddit = 'worldnews'

# logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def file_title(raw_title, file_prefix='', file_extension='.txt', max_characters=20):

	# Cap Length / Generate Short Title
	if len(raw_title) > max_characters:
		raw_title = extract(raw_title)[0]

	# Format File Title

	# Remove all non-word characters (everything except numbers and letters)
	free_title = re.sub(r"[^\w\s]", '', raw_title)

	# Replace all runs of whitespace with a single dash
	file_title = file_prefix + '_' + re.sub(r"\s+", '_', free_title) + file_extension

	return file_title


def save_subreddit_to_dir(subreddit, directory):
	logger.info('Connecting to subreddit: %s' % subreddit)
	r = requests.get(
	    'http://www.reddit.com/r/{}.json'.format(subreddit),
	    headers={'user-agent': 'Mozilla/5.0'}
	)

	for post in r.json()['data']['children']:
		try:
			title = post['data']['title']
			url = post['data']['url']
			fname = file_title(title, subreddit)

			article = Article(url)
			article.download()
			article.parse()
			text = article.text

			f = {'title': title, 'url':url, 'text':text}

			logger.info('Creating File: %s' % (directory + fname))
			with open(directory + fname, 'w+') as outfile:
				json.dump(f, outfile)
		except:
			logger.debug('failure in save_subreddit_to_dir')

	logger.info('Subreddit Scrapping Complete.')

if __name__ == '__main__':
	save_subreddit_to_dir(subreddit, path)

