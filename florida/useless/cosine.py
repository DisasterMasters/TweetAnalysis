import csv 
import os
import re
import random
import sys
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

topics = []

#open file with either nmf or lda topics, put into array of arrays
file = open(sys.argv[1], "r")
for line in file:
	match = re.match(r'^Topic', line)	
	if not match:
		top = line.strip().split(" ")
		topics.append(top)

tweets = []

#put tweets into one large array
with open("old/twits.txt", "r") as tops: 
	for line in tops:
		tweets = line.strip().split('\t') #split all tweets into array


output_name = str(sys.argv[1])
output_name = output_name.split(".")[0].upper() + "_cosine"
file = open("old/newtweets.txt", "w")

for t in topics:
	t = ' '.join(t)
	file.write("\n\n***********************************************\n\n")
	file.write("Topic: \n\n")
	file.write(t)
	output = []
	for tweet in tweets:
		if len(output) == 100:
			break
		topic = [t, tweet]
		tfidf_vectorizer = TfidfVectorizer()
		topic_matrix = tfidf_vectorizer.fit_transform(topic)
		cosine = cosine_similarity(topic_matrix[0:1], topic_matrix[1:2])
		if cosine >= float(sys.argv[2]): #calibrate the sensitivity from command line
			if len(tweet) > 20:
				output.append(tweet)
	file.write("\n\nSample Tweets:\n\n")
	for i in range(0, len(output)):
		file.write("\n\n")
		try:
			file.write(output[i]) #for if there arent enough tweets to fill 20
		except Exception as e:
			print e
file.close()	

