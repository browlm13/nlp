#!/usr/bin/env python

"""
RAKE and Sentence Ranking

sources : 
https://github.com/csurfer/rake-nltk/blob/master/rake_nltk/rake.py,
https://gist.github.com/alexbowe/879414

papers:
	1.) title: Automatic keyphrase extraction from scientific articles
		authors: Su Nam Kim • Olena Medelyan • Min-Yen Kan • Timothy Baldwin

"""
#internal
import string
import operator
import re

#external
import numpy as np
import nltk
from nltk.tokenize import RegexpTokenizer
from nltk import FreqDist
from nltk.stem.porter import *
from nltk.corpus import stopwords
from nltk.tag import pos_tag


"""
	Rapid Automatic Keyword Extraction
"""

def is_punctuation(word):
	return len(word) == 1 and word in string.punctuation #or re.search('[^\w\d\s\-\_]{2}', word)

def is_stopword(word):
	stop_words = nltk.corpus.stopwords.words()
	return word in stop_words

def isNumeric(word):
	try:
		float(word) if '.' in word else int(word)
		return True
	except ValueError:
		return False

"""
def generate_candidate_keywords(sentences):
	stop_words = nltk.corpus.stopwords.words()

	phrase_list = []
	for sentence in sentences:
	  words = map(lambda x: "|" if x in stop_words else x,
		nltk.word_tokenize(sentence.lower()))
	  phrase = []
	  for word in words:
		if word == "|" or isPunct(word):
		  if len(phrase) > 0:
			phrase_list.append(phrase)
			phrase = []
		else:
		  phrase.append(word)

	return phrase_list
"""

def break_word(word, marker):
	""" seperate phrases based on break_words, return marker if word is break_word, otherwise return word """

	if is_stopword(word) or is_punctuation(word):
		return marker

	return word

def generate_candidate_keywords(sentences):
	marker = "|"

	phrase_list = []
	for sentence in sentences:
		sentence = nltk.word_tokenize(sentence.lower())
		words = [break_word(w,marker) for w in sentence]
		
		phrase = []
		for word in words:
			if (word != marker):
				phrase.append(word)
			elif len(phrase) > 0:
				phrase_list.append(phrase)
				phrase = []

	return phrase_list

def calculate_phrase_scores(phrase_list, word_scores):
	""" Phrase score is equal to sum of word scores contained in phrase. """

	phrase_scores = {}
	for phrase in phrase_list:
		phrase_score = 0
		for word in phrase:
			phrase_score += word_scores[word]
		phrase_scores[" ".join(phrase)] = phrase_score
	return phrase_scores

def calculate_word_scores(phrase_list):
	word_freq = nltk.FreqDist()
	word_degree = nltk.FreqDist()
	for phrase in phrase_list:
		degree = len(list(filter(lambda x: not isNumeric(x), phrase))) - 1
		for word in phrase:
			word_freq[word] += 1
			word_degree[word] += degree
	for word in word_freq.keys():
		word_degree[word] = word_degree[word] + word_freq[word] # itself
	# word score = deg(w) / freq(w)
	word_scores = {}
	for word in word_freq.keys():
		word_scores[word] = word_degree[word] / word_freq[word]
	return word_scores

def extract(text):
	sentences = nltk.sent_tokenize(text)

	phrase_list = generate_candidate_keywords(sentences)
	word_scores = calculate_word_scores(phrase_list)
	phrase_scores = calculate_phrase_scores(phrase_list, word_scores)

	sorted_phrase_scores = sorted(phrase_scores.items(), key=operator.itemgetter(1), reverse=True)
	n_phrases = len(sorted_phrase_scores)

	top_fraction = 1 #1/3
	return list(map(lambda x: x[0],sorted_phrase_scores[0:int(n_phrases/top_fraction)]))

def top_words(text):
	sentences = nltk.sent_tokenize(text)

	phrase_list = generate_candidate_keywords(sentences)
	word_scores = calculate_word_scores(phrase_list)
	sorted_word_scores = sorted(word_scores.items(), key=operator.itemgetter(1), reverse=True)
	
	top_picks = [ws[0] for ws in sorted_word_scores]
	return list(top_picks)

"""
	rank sentences based on rank top phrases
"""
def rank_sentences(text, max_sentences=None):
	tokenizer = RegexpTokenizer(r'\w+')
	sentences = nltk.sent_tokenize(text)
	top_phrases = extract(text)
	top_sentences = []

	print(top_phrases)

	#max sentences
	#if max_sentences is None:
	#	max_sentences = max(len(top_phrases), len(sentences))
	#max_sentences = min(len(top_phrases), len(sentences), max_sentences)

	for i in range(len(top_phrases)):
		tokenized_phrase = tokenizer.tokenize(top_phrases[i])
		for s in sentences:

			#print("\nphrase:%s,\n\nsentence:%s\n\n\n" % (str(set(tokenized_phrase)),str(set(tokenizer.tokenize(s.lower())) )))
			
			if (set(tokenized_phrase) < set(tokenizer.tokenize(s.lower()))) and (s not in top_sentences):
				top_sentences.append(s)
				
				print(s)
	return top_sentences

