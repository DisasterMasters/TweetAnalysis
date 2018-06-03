import csv 
import os
import re
import random
import sys

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
with open("training_data/twits.txt", "r") as tops: 
	for line in tops:
		tweets = line.strip().split('\t') #split all tweets into array

output_name = str(sys.argv[1])
output_name = output_name.split(".")[0].upper() + "_output"
file = open("results/"+output_name+".txt", "w")
for top in topics: #iterate over each topic as whole
	file.write("\n\n***********************************************\n\n")
	file.write("Topic: \n\n")
	file.write(' '.join(top))
	output = []
	for tweet in tweets: #iterate through each tweet underneath one topic
		wordcount = 0
		for t in top: #iterate through each individual word pertaining to one topic
			if t.lower() in tweet.lower(): #check to see if word is in tweet
				wordcount += 1
		comp = 3
		if str(sys.argv[1]) == "nmf.txt":
			comp = 4
		if wordcount >= comp: #need to have words over a certain threshold
			output.append(tweet)
	file.write("\n\nSample Tweets:\n\n")
	for i in range(0, 25):
		file.write("\n\n")
		file.write(random.choice(output))

file.close()

			
				

