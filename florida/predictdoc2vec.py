import csv
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import itertools
from datetime import datetime
import re
import math
import random
from itertools import repeat
import sys
import os
from gensim import utils
from gensim.models import Doc2Vec
import gensim
import numpy as np
import nltk
import io

typeoffile = sys.argv[1]
included = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]

file_name = ''
if typeoffile == 'utility':
        file_name = 'training_data/utility_data.txt'
        included = [1, 4, 8, 9, 10, 11, 12, 14, 15]
if typeoffile == 'media':
        file_name = 'training_data/media_data.txt'
if typeoffile == 'nonprofit':
        file_name = 'training_data/nonprofit_data.txt'
if typeoffile == 'gov':
        file_name = 'training_data/gov_data.txt'



model = Doc2Vec.load('doc2vecmodels/doc2vecmodel_' + typeoffile)
#get the test data from the dataset of tweets
test_data = []


#open test data obtained from media/utility file, parse
file = open(file_name)
w = file.read()
test = w.split("\t")
for t in test:
	if t != '\n':
		t = t.split('~+&$!sep779++')
		if t != ['']:
			test_data.append([t[0], t[1], t[2]])

		
#open predictions file for writing
outfile = open("results/" + typeoffile + "_supervised_doc2vec.csv", "w")
writer = csv.writer(outfile)
writer.writerow(['Tweet', 'Category', 'Date', 'Permalink'])


#make prediction for each tweet, write to file
for t in test_data:
	split = t[0].decode('utf-8')
	split = nltk.word_tokenize(split)
        vect = model.infer_vector(split)
	#get cosine similarity, keep highest score
	result = model.docvecs.most_similar(positive=[vect], topn=1)
	prediction = result[0][0]
	#print prediction
	if int(prediction) in included:
		date = ''
                #clean up mixed date formats -__-
                if '/' in t[1]:
                        date = datetime.strptime(t[1], '%m/%d/%Y')
                elif '-' in t[1]:
                        date = datetime.strptime(t[1], '%Y-%m-%d')
		#writing tweet, prediction, date, and permalink
		writer.writerow([t[0], int(prediction), date.strftime('%m/%d/%Y'), t[2]])
	

