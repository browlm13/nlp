#!/usr/bin/env python

"""
	Normalize and clean text methods
"""
import logging
import string
import math

import nltk
from nltk.tokenize import RegexpTokenizer
from nltk.stem import PorterStemmer
from nltk import FreqDist
from nltk.tag import pos_tag


# logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def is_stopword(word):
	""" Return True of word is in stop word list """
	stop_words = nltk.corpus.stopwords.words('english')
	return word in stop_words

def is_punctuation(word):
	return len(word) == 1 and word in string.punctuation

def is_number(word):
	try:
		float(word)
		return True
	except ValueError:
		logger.debug('ValueError is_number')
 
	try:
		import unicodedata
		unicodedata.numeric(word)
		return True
	except (TypeError, ValueError):
		logger.debug('ValueError is_number')
	 
	return False

def is_shorter(word,n=3):
	if len(word) < n:
		return True
	return False

def stem(word):
	ps = PorterStemmer()
	return ps.stem(word)


def clean_word(raw_word):
	raw_word = raw_word.lower()
	if is_stopword(raw_word) or is_punctuation(raw_word) or is_shorter(raw_word) or is_number(raw_word):
		word = ""
	else:
		word = stem(raw_word)
	return word


def set_clean_raw_text(raw_text):
	""" tokenize sentence, convert to lower, stem, remove stop words, numbers, punctuation"""
	logger.debug('Cleaning Text')

	#tokenize and lower sentence
	tokenizer = RegexpTokenizer(r'\w+')
	tokens = tokenizer.tokenize(raw_text.lower())		# tokens = nltk.word_tokenize(corpus.lower()) # without removing punctiation

	#remove stop words
	tokens = [w for w in tokens if not is_stopword(w)]

	#remove punctuation
	tokens = [w for w in tokens if not is_punctuation(w)]

	#remove short 
	tokens = [w for w in tokens if not is_shorter(w)]

	#remove number
	tokens = [w for w in tokens if not is_number(w)]

	#stem words
	tokens = map(stem, tokens)

	logger.debug('Cleaning Text Complete')
	return set(tokens)

def set_clean_tokens(raw_token_list):
	""" clean tokenized sentence, convert to lower, stem, remove stop words, numbers, punctuation"""
	logger.debug('Cleaning Text')

	clean_tokens = []
	for t in raw_token_list:
		clean_token = clean_word(t)
		if clean_token != "":
			clean_tokens.append(clean_token)

	return set(clean_tokens)

def processes_and_tokenize(raw_document):
	""" remove punctuation, convert to lower case, and return list of tokens """
	tokenizer = RegexpTokenizer(r'\w+')
	tokens = tokenizer.tokenize(raw_document.lower())		# tokens = nltk.word_tokenize(corpus.lower()) # without removing punctiation

	#remove stop words
	stop_words = set(nltk.corpus.stopwords.words('english'))
	#stop_words = set(stopwords.words('english'))
	filtered_tokens = [w for w in tokens if not w in stop_words]
	return filtered_tokens


def word_frequency_dict(tokens):
	""" returns a dictionary of word and their assosiated frequencies from token list """

	fdist = FreqDist(tokens) 						# fdist.keys() fdist.values()
	return dict(fdist)

def term_fequency(term,tokens):
	term = processes_and_tokenize(term)[0]	#make sure term is in correct form

	tf = tokens.count(term)
	return tf/len(tokens)

def augmented_term_fequency(term,tokens):
	""" returns term frequency in tokens over maximum term frequency of tokens """
	term = processes_and_tokenize(term)[0] #make sure term is in correct form

	max_count = max([tokens.count(t) for t in tokens])
	return tokens.count(term)/max_count

def inverse_document_frequency(term, tokenized_documents_list):
	""" IDF(t) = ln( Number Of Documents / Number Of Documents Containg Term )."""
	term = processes_and_tokenize(term)[0]	#make sure term is in correct form

	num_documents = len(tokenized_documents_list)
	num_documents_with_term = len([document for document in tokenized_documents_list if term in document])
	
	assert num_documents_with_term > 0
	return math.log(num_documents / num_documents_with_term)


def nolog_inverse_document_frequency(term, tokenized_documents_list):
	""" IDF(t) = ln( Number Of Documents / Number Of Documents Containg Term )."""
	term = processes_and_tokenize(term)[0]	#make sure term is in correct form

	num_documents = len(tokenized_documents_list)
	num_documents_with_term = len([document for document in tokenized_documents_list if term in document])
	
	assert num_documents_with_term > 0
	return num_documents / num_documents_with_term

def tf_idf(term, tokenized_document, tokenized_documents_list):
	""" Term Frequency - Inverse Document Frequency : returns tf * idf """
	#return term_fequency(term, tokenized_document) * inverse_document_frequency(term, tokenized_documents_list)
	#return augmented_term_fequency(term, tokenized_document) * inverse_document_frequency(term, tokenized_documents_list)
	return term_fequency(term, tokenized_document) * nolog_inverse_document_frequency(term, tokenized_documents_list)


def keyword_score(term, tokenized_document, tokenized_documents_list):
	tf_idf_scaler = 2
	term_tf_idf_score = tf_idf(term,tokenized_document,tokenized_documents_list)
	term_similarity_score = similarity_score(term, tokenized_document)
	return tf_idf_scaler*term_tf_idf_score + term_similarity_score


def keyword_scores_for_part_of_speech(pos, tokenized_document, tokenized_documents_list):
	tagged_tokenized_document = pos_tag(tokenized_document)
	filtered_tokens = [term for term, tag in tagged_tokenized_document if tag == pos]
	return list(filtered_tokens)

test_crazy = """
	U.S.A. is the best-ish country in the world. I'm supprised how far $12.05 pieces of lake 
	from... The Buddha, the Godhead, resides quite as comfortably in the circuits of a digital
	computer or the gears of a cycle transmission as he does at the top of a mountain
	or in the petals of a flower. To think otherwise is to demean the Buddha...which is
	to demean oneself. L.A. I'm a fiagol! Are you? P.S. I love you,
	Megan. 12:00 a.m. , 11:23 PM tonight. 12 o'clock

	It's 12:00 am and its time for a bath. There are 100,000 yellow men on a train to calcuta.
	 and 2,000,560,786 jews. 5,000 in my ear. My savings 3.12$.

	 2,500,766.30$
	 $2,500,766.333 
	My mother yerns the chocolate rain. L.J. is amazing e.g. great
	tammy-lynn is a whore. you and I. YOU and I, we are both tierd.

	People in England who commit the most serious crimes of 
	animal cruelty could face up to five years in prison, the government has said.
	The move - an increase on the current six-month maximum sentence - follows a 
	number of cases where English courts wanted to hand down tougher sentences.
	Environment Secretary Michael Gove said it would target "those who commit the 
	most shocking cruelty towards animals".
	The RSPCA said it would "deter people from abusing and neglecting animals"."
"""

#print(set_clean_tokenize(test_crazy))

