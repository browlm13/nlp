#python 3

"""
	Load subreddit articles
"""

#internal
import json
import os
import logging

# logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

#
#	Data Set Methods
#

def load_scraped_subreddit_document_set(path, subreddit, titles=False):
	""" loads all json files from dataset matching subreddit format: "subreddit_fname.extension" and returns list of ['title','text'] """
	logger.info('Loading Document Set')

	files = [f for f in os.listdir(path) if f.split('_')[0] == subreddit]
	documents = []
	for f in files:
		with open(path + f) as data_file:    
			data = json.load(data_file)

		"""
		if titles == True:
			documents.append([data['title'],data['text']])
		else:
			documents.append(data['text'])
		"""
		if titles == True:
			document = data
		else:
			document = data['text']
		documents.append(document)

	logger.info('Document Set Loading Complete')
	return documents

def get_document_jsons_list(path = "data_set/",  subreddit = "worldnews"):
	all_documents = load_scraped_subreddit_document_set(path,subreddit, True)
	return all_documents