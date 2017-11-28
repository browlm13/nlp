#python 3

import normalizers as norm

"""
similarity measures

Jaccard index / Intersection over Union / Jaccard similarity coefficient

Overlap Coefficient  - the overlap between two sets, and is defined as the size of the intersection divided by the smaller of the size of the two sets:
"""

def jaccard_similarity_coefficient(a, b):
	""" compute jaccard index after processing (tokenize sentence, convert to lower, stem, remove stop words, numbers, punctuation)"""
	a_words, b_words = map(norm.set_clean_tokens, [a,b])

	intersection = set.intersection(a_words, b_words)
	union = set.union(a_words, b_words)

	#try to compute jaccard index
	try: jaccard_index = len(intersection)/len(union)
	except: pass

	#empty sets
	if len(a_words) == 0 or len(b_words) == 0:
		jaccard_index = 0

	return jaccard_index

def overlap_coefficient(a, b):
	""" compute overlap_coefficient after processing (tokenize sentence, convert to lower, stem, remove stop words, numbers, punctuation)"""
	a_words, b_words = map(norm.set_clean_tokens, [a,b])

	intersection = set.intersection(a_words, b_words)
	length_a_words, length_b_words = len(a_words), len(b_words)

	#empty sets
	if length_a_words == 0 or length_b_words == 0: return 0

	#try to compute overlap_coefficient
	try: overlap_coefficient = len(intersection)/min(length_a_words,length_b_words)
	except: overlap_coefficient = 0

	return overlap_coefficient


from difflib import SequenceMatcher
def similar(a, b):
    return SequenceMatcher(None, a, b).ratio()

# similarity between character similarity
def total_char_similarity(a,b):
	""" compute similarity score of characters for each word in cartesian product"""
	a_words, b_words = map(norm.set_clean_tokens, [a,b])

	total_score = 0
	for ai in a_words:
		for bi in b_words:
			total_score += similar(ai, bi)
	return total_score

def similarity_score(a,b):
	""" combine all similarity measures into single score """
	jsc_scaler = 15
	ocs_scaler = 5
	tcss_scaler = 0.05

	jaccard_similarity_coefficient_score = jsc_scaler * jaccard_similarity_coefficient(a,b)
	overlap_coefficient_score = ocs_scaler * overlap_coefficient(a,b)
	total_char_similarity_score = tcss_scaler * total_char_similarity(a,b)
	total_score = jaccard_similarity_coefficient_score + overlap_coefficient_score + total_char_similarity_score
	
	return total_score

