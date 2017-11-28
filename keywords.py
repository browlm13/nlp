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

import similarity_measures as sim

#external
import numpy as np
import nltk
from nltk.tokenize import RegexpTokenizer
from nltk.stem.porter import *
from nltk.corpus import stopwords
from nltk.tag import pos_tag
from nltk import FreqDist

def remove_scores(list_of_tupples):
	return [i[0] for i in list_of_tupples]

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
			try: phrase_scores[phrase] += word_scores[word]
			except: pass

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

def top_words(document, n=5, return_scores=True):
	""" Return top n phrases after computing scores """
	phrase_list = extract_phrases(document)
	word_scores = compute_word_scores(phrase_list)

	n = min(len(word_scores), n)
	top_words = sorted(word_scores.items(), key=operator.itemgetter(1), reverse=True)[:n]

	if return_scores == False:
		return remove_scores(top_words)
	return top_words
	

def top_phrases(document, n=5, return_scores=True):
	""" Return top n phrases after computing scores """
	phrase_list = extract_phrases(document)
	phrase_scores = compute_phrase_scores(phrase_list)

	n = min(len(phrase_list), n)
	top_phrases = sorted(phrase_scores.items(), key=operator.itemgetter(1), reverse=True)[:n]

	if return_scores == False:
		return remove_scores(top_phrases)
	return top_phrases

def top_sentences(document, n=2, return_scores=True, original_order=False):
	""" Return top n sentences after computing scores """

	sentence_list = extract_sentences(document)
	sentence_scores = compute_sentence_scores(document)

	n = min(len(sentence_list), n)
	top_sentences = sorted(sentence_scores.items(), key=operator.itemgetter(1), reverse=True)[:n]


	#if original order is selected:
	threshold = 0.8 #similarity threshold
	min_length = 3 # words
	ordered_top_sentences = []
	if original_order == True:
		no_score_top_sentences = remove_scores(top_sentences)
		for original_sentence in sentence_list:
			for top_sentence in no_score_top_sentences:
				if sim.similar(original_sentence,top_sentence) > threshold:
					num_words = len(original_sentence.split(' '))
					if num_words > min_length:
						ordered_top_sentences.append(original_sentence)
		return ordered_top_sentences

	if return_scores == False:
		return remove_scores(top_sentences)
	return top_sentences

def summary(document, percentage):
	all_sentences = extract_sentences(document)
	num_sentence_to_extract= int(percentage* len(all_sentences))
	key_sentence_list = top_sentences(document,n=num_sentence_to_extract, return_scores=False, original_order=True)
	reduced_summary = ' '.join(key_sentence_list)
	return reduced_summary

#
#	testing
#
text = """Jared Kushner made a secret visit to Saudi Arabia last week with other officials from President Donald Trump’s administration for talks on peace in the Middle East.

The president’s son-in-law and senior adviser made what was his third trip to the Gulf kingdom this year alone, Politico reported, citing a White House official.

Kushner traveled commercially on Wednesday and was in the country with Trump’s Middle East envoy Jason Greenblatt and deputy national security adviser Dina Powell. It remains unclear whom they met with on their trip.
The visit was part of an ongoing regional effort by the White House to draw the influential powers in the Middle East into helping with the peace process between Israel and the Palestinians, a decades-long conflict in which negotiations have remained dormant for years.

“While these regional talks will play an important role, the president reaffirms that peace between Israelis and Palestinians can only be negotiated directly between the two parties and that the United States will continue working closely with the parties to make progress toward that goal,” the White House official told Politico in a statement.

“No deal will be imposed on Israelis and Palestinians. We are committed to facilitating a deal that improves conditions for both parties.”
Trump has tasked Kushner with sowing the seeds of peace in the Middle East, and Kushner was instrumental to a $110 billion arms deal signed between Washington and Riyadh in May, when Trump and a presidential delegation paid a visit to the country.

Both Kushner and Greenblatt have embarked on a period of shuttle diplomacy across the Middle East, from Amman to Jerusalem, Ramallah, Dubai and Cairo, since Trump’s January 20 inauguration.

To that end, the US is trying to prevent Israel’s right wing from taking unilateral measures that could harm any chance of reviving peace negotiations, which have been moribund since they last collapsed in 2014.

Israeli lawmakers were set to vote on legislation for a “Greater Jerusalem” on Sunday, in which several West Bank settlements would come under the responsibility of the Jerusalem municipality. The legislation was set to pass and set off the process of ratification. The international community deems the settlements, built on land Palestinians have earmarked for any future sovereign state, to be illegal.

But the bill was delayed after US pressure for fear that it would hamper any hopes of peace with the Palestinians, according to right-wing lawmaker David Bitan, chairman of Israeli Prime Minister Benjamin Netanyahu’s ruling coalition.

“There is American pressure that claims this is about annexation and that this could interfere with the peace process,” Bitan told Army Radio.

“The prime minister doesn’t think this is about annexation. I don’t think so either. We have to take the time to clarify matters to the Americans. Therefore, if the bill passes in a week, or in a month, it’s less problematic,” he said.

"""

#print(top_phrases(text))
#print(top_words(text))
#print(top_sentences(text))
