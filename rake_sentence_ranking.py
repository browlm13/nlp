#!/usr/bin/env python

"""
RAKE and Sentence Ranking

code base : https://github.com/csurfer/rake-nltk/blob/master/rake_nltk/rake.py

"""
#internal
import string
import operator

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

def isPunct(word):
  return len(word) == 1 and word in string.punctuation

def isNumeric(word):
	try:
		float(word) if '.' in word else int(word)
		return True
	except ValueError:
		return False

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

def calculate_phrase_scores(phrase_list, word_scores):
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
def rank_sentences(text, n):
	tokenizer = RegexpTokenizer(r'\w+')
	sentences = nltk.sent_tokenize(text)
	top_phrases = extract(text)
	top_sentences = []

	for i in range(n):
		tokenized_phrase = tokenizer.tokenize(top_phrases[i])
		for s in sentences:
			if (set(tokenized_phrase) < set(tokenizer.tokenize(s))) and (s not in top_sentences):
				top_sentences.append(s)
	return top_sentences