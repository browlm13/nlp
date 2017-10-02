#!/usr/bin/env python

"""
Keyword Extraction

sources : 
https://github.com/csurfer/rake-nltk/blob/master/rake_nltk/rake.py,
https://gist.github.com/alexbowe/879414
http://alexbowe.com/au-naturale/

papers:
	1.) title: Automatic keyphrase extraction from scientific articles
		authors: Su Nam Kim • Olena Medelyan • Min-Yen Kan • Timothy Baldwin
"""

#internal
import string
import operator
import re
import copy

#external
import numpy as np
import nltk
from nltk.tokenize import RegexpTokenizer
from nltk.stem.porter import *
from nltk.corpus import stopwords
from nltk.tag import pos_tag
from nltk import FreqDist


def is_punctuation(word):
	""" Return True if word is composed entirly of punctuation and whitespace """
	if set(word) < set(string.punctuation + string.whitespace):
		return True
	return False

def is_stopword(word):
	""" Return True of word is in stop word list """
	stop_words = nltk.corpus.stopwords.words()
	return word in stop_words

def break_word(word, marker):
	""" Seperate phrases based on break_words, return marker if word is break_word, otherwise return word """
	if is_stopword(word) or is_punctuation(word):
		return marker
	return word

def extract_sentences(document):
	sentences = nltk.sent_tokenize(document.lower())
	return sentences

def extract_words(sentence):
	words = nltk.word_tokenize(sentence)
	return words

def extract_phrases(document):
	""" Exctract non stop words and phrases from document and return list of words and phrases"""
	marker = '|'
	sentences = extract_sentences(document)

	phrase_list = []
	for sentence in sentences:
		sentence = extract_words(sentence)
		phrase_list += [break_word(w,marker) for w in sentence]

	phrase_list = ' '.join(phrase_list).split(marker)
	return list([p.strip() for p in phrase_list if is_punctuation(p) == False])

def compute_word_scores(phrase_list):
	""" Return dictionary of word scores. word_score = deg(w) / freq(w),
	where freq(w) is the number of document occurences, 
	and where deg(w) is the number of document occurences 
	plus the combined word length of its member phrases"""

	tokens = extract_words(' '.join(phrase_list))
	word_freq = FreqDist(tokens)
	word_degree = FreqDist(tokens)

	for phrase in phrase_list:
		for word in extract_words(phrase):
			word_degree[word] += len(phrase)-1

	word_scores = {}
	for word in word_freq.keys():
		word_scores[word] = word_degree[word] / word_freq[word]

	return word_scores

def compute_phrase_scores(phrase_list):
	""" Phrase score is equal to sum of word scores contained in phrase """
	word_scores = compute_word_scores(phrase_list)

	phrase_scores = {p:0 for p in phrase_list}
	for phrase in phrase_list:
		for word in extract_words(phrase):
			phrase_scores[phrase] += word_scores[word]
	return phrase_scores

def compute_sentence_scores(document):
	""" Sentence score is equal to sum of word and phrase scores contained in sentence """

	word_list = extract_words(document)
	phrase_list = extract_phrases(document)	
	sentence_list = extract_sentences(document)

	word_scores = compute_word_scores(phrase_list)
	phrase_scores = compute_phrase_scores(phrase_list)

	sentence_scores = {s:0 for s in sentence_list}
	for sentence in sentence_list:
		for word in extract_words(' '.join(phrase_list)):
			sentence_scores[sentence] += word_scores[word]

	for sentence in sentence_list:
		for phrase in phrase_list:
			if set(phrase) < set(sentence):
				sentence_scores[sentence] += phrase_scores[phrase]

	return sentence_scores

def top_words(document, n=5):
	""" Return top n phrases after computing scores """
	phrase_list = extract_phrases(document)
	word_scores = compute_word_scores(phrase_list)

	n = min(len(word_scores), n)
	return sorted(word_scores.items(), key=operator.itemgetter(1), reverse=True)[:n]

def top_phrases(document, n=5):
	""" Return top n phrases after computing scores """
	phrase_list = extract_phrases(document)
	phrase_scores = compute_phrase_scores(phrase_list)

	n = min(len(phrase_list), n)
	return sorted(phrase_scores.items(), key=operator.itemgetter(1), reverse=True)[:n]

def top_sentences(document, n=2):
	""" Return top n sentences after computing scores """

	sentence_list = extract_sentences(document)
	sentence_scores = compute_sentence_scores(document)

	n = min(len(sentence_list), n)
	return sorted(sentence_scores.items(), key=operator.itemgetter(1), reverse=True)[:n]


#
#	testing
#
text = """People in England who commit the most serious crimes of 
animal cruelty could face up to five years in prison, the government has said.
The move - an increase on the current six-month maximum sentence - follows a 
number of cases where English courts wanted to hand down tougher sentences. Especially against tiger. tiger tiger tiger.
Environment Secretary Michael Gove said it would target "those who commit the 
most shocking cruelty towards animals". 23 people died today. 24 will have died tommorow. Mike Tyson fights tiger with a broom.
The RSPCA said it would "deter people from abusing and neglecting animals"."""


print(top_phrases(text))
print(top_words(text))
print(top_sentences(text))
