#!/usr/bin/env python

"""
	NLP Summary
"""

#internal
import math
import json
import os
import logging

# mylib
from rake_sentence_ranking import *

#external
import numpy as np
import nltk
from nltk.tokenize import RegexpTokenizer
from nltk import FreqDist
from nltk.stem.porter import *
from nltk.corpus import stopwords
from nltk.tag import pos_tag
from nltk.corpus import wordnet

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

		if titles == True:
			documents.append([data['title'],data['text']])
		else:
			documents.append(data['text'])

	logger.info('Document Set Loading Complete')
	return documents

#
#	Wordnet Methods
#

def syns_tag(term):
	""" returns word wordnet.synset """
	try:
		syns = wordnet.synsets(term)
		name = syns[0].name()
		return wordnet.synset(name)
	except:
		return None

def similarity_score(term, tokenized_document):
	""" Calculates similarity score for word against all other words in document """
	term = processes_and_tokenize(term)[0]	#make sure term is in correct form
	
	#minimum score
	minimum_score = .7

	# format words to wordnet.synset
	main_tagged_word = syns_tag(term)

	if main_tagged_word is not None:
		tokenized_tagged_document = []
		for t in tokenized_document:
			try:
				tokenized_tagged_document.append(syns_tag(t))
			except: pass

		# calculate similarity score
		similarity_scores = []
		for t in tokenized_tagged_document:
			try:
				ss = main_tagged_word.wup_similarity(t)
			except:
				ss = None
			if ss is not None and type(ss) :
				if ss > minimum_score:
					similarity_scores.append(ss)
		summed_similarity_scores = sum(similarity_scores)

		assert len(tokenized_tagged_document) > 0
		score = summed_similarity_scores / len(tokenized_tagged_document)
		return score

	else: return 0

#
#	Processing Methods
#

def processes_and_tokenize(raw_document):
	""" remove punctuation, convert to lower case, and return list of tokens """
	logger.info('Cleaning Text')

	tokenizer = RegexpTokenizer(r'\w+')
	tokens = tokenizer.tokenize(raw_document.lower())		# tokens = nltk.word_tokenize(corpus.lower()) # without removing punctiation

	#remove stop words
	stop_words = set(stopwords.words('english'))
	filtered_tokens = [w for w in tokens if not w in stop_words]

	logger.info('Cleaning Text Complete')
	return filtered_tokens

#
#	Term Frequency Methods
#

def word_frequency_dict(tokens):
	""" returns a dictionary of word and their assosiated frequencies from token list """
	logger.info('Building Word Frequency Dictionary')

	fdist = FreqDist(tokens) 						# fdist.keys() fdist.values()
	logger.info('Word Frequency Dictionary Completed')

	return dict(fdist)

def term_fequency(term,tokens):
	""" Return term frequency / number of terms in token list """
	logger.info('Calculating Term Frequency')

	term = processes_and_tokenize(term)[0]	#make sure term is in correct form

	tf = tokens.count(term)
	return tf/len(tokens)

def augmented_term_fequency(term,tokens):
	""" returns term frequency in tokens over maximum term frequency of tokens """
	logger.info('Calculating Augmented Term Frequency')

	term = processes_and_tokenize(term)[0] #make sure term is in correct form

	max_count = max([tokens.count(t) for t in tokens])
	return tokens.count(term)/max_count

def inverse_document_frequency(term, tokenized_documents_list):
	""" IDF(t) = ln( Number Of Documents / Number Of Documents Containg Term )."""
	logger.info('Calculating IDF')

	term = processes_and_tokenize(term)[0]	#make sure term is in correct form

	num_documents = len(tokenized_documents_list)
	num_documents_with_term = len([document for document in tokenized_documents_list if term in document])
	
	assert num_documents_with_term > 0
	return math.log(num_documents / num_documents_with_term)


def nolog_inverse_document_frequency(term, tokenized_documents_list):
	""" IDF(t) = ln( Number Of Documents / Number Of Documents Containg Term )."""
	logger.info('Calculating no-log IDF')

	term = processes_and_tokenize(term)[0]	#make sure term is in correct form

	num_documents = len(tokenized_documents_list)
	num_documents_with_term = len([document for document in tokenized_documents_list if term in document])
	
	assert num_documents_with_term > 0
	return num_documents / num_documents_with_term

def tf_idf(term, tokenized_document, tokenized_documents_list):
	""" Term Frequency - Inverse Document Frequency : returns tf * idf """
	logger.info('Calculating TF-IDF')

	#return term_fequency(term, tokenized_document) * inverse_document_frequency(term, tokenized_documents_list)
	#return augmented_term_fequency(term, tokenized_document) * inverse_document_frequency(term, tokenized_documents_list)
	return term_fequency(term, tokenized_document) * nolog_inverse_document_frequency(term, tokenized_documents_list)

#
#	Scoring Methods
#

def keyword_score(term, tokenized_document, tokenized_documents_list):
	logger.info('Calculating Keyword Score')

	tf_idf_scaler = 2
	term_tf_idf_score = tf_idf(term,tokenized_document,tokenized_documents_list)
	term_similarity_score = similarity_score(term, tokenized_document)
	return tf_idf_scaler*term_tf_idf_score + term_similarity_score


def keyword_scores_for_part_of_speech(pos, tokenized_document, tokenized_documents_list):
	logger.info('Calculating Keyword Score For Part Of Speech')

	tagged_tokenized_document = pos_tag(tokenized_document)
	filtered_tokens = [term for term, tag in tagged_tokenized_document if tag == pos]
	return list(filtered_tokens)


#
#  _____ ___ ___ _____ ___ _  _  ___ 
# |_   _| __/ __|_   _|_ _| \| |/ __|
#   | | | _|\__ \ | |  | || .` | (_ |
#   |_| |___|___/ |_| |___|_|\_|\___|
#


#
#			Load Data Set
#

data_set_path = 'data_set/'
subreddit = 'worldnews'

all_documents = load_scraped_subreddit_document_set(data_set_path, subreddit)
tokenized_documents_list = list(map(processes_and_tokenize, all_documents))

#
#			Testing Score Methods
#

all_document_keyword_scores = []
for d in tokenized_documents_list:
	document_keyword_scores = []
	for t in d:
		#print ("word: %s,\t tf: %f,\t idf: %f,\t tf-idf: %f,\t similarity_score: %f" % (t, term_fequency(t,d), inverse_document_frequency(t,tokenized_documents_list), tf_idf(t,d,tokenized_documents_list), similarity_score(t,d)))
		document_keyword_scores.append((t,keyword_score(t,d,tokenized_documents_list)))
	document_keyword_scores.sort(key=lambda x: x[1])
	all_document_keyword_scores.append(document_keyword_scores)

for d in all_document_keyword_scores:
	print ("\n\n\n")
	for i in d:
		print ("term: %s,\t\t keyword_score: %f" % (i[0], i[1]))
