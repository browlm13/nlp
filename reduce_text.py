#python 3

"""
	remove lowest scoring sentences from article to reduce size
"""

#internal
import json
import os
import logging

#mylibs
import load_documents
import normalizers as norm
import keywords as key
import similarity_measures as sim

# logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

all_documents = load_documents.get_document_jsons_list()

d = all_documents[0]

p = 0.3

print(key.summary(d['text'],p))