import sys
import re
import csv
import operator
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity



docs = []
with open("training_data/dates.txt") as file:
        for line in file:
                docs = line.split("\t")

def getIndices(full_categories, selections):
        indices = []
        s = selections.read().splitlines()
        clean = []
        for line in full_categories:
                match = re.match(r'^Topic', line)
                if not match:
                        line = line.strip()
                        clean.append(line)
        select = []

        for k, v in enumerate(clean):
                if v in s:
                        select.append(v)
                        indices.append(k)

        return indices, select


def getTweets(indices, weights, title):
        weights = weights.read()
        regex = r"\[(.*?)\]"
        split_list = []
        weight_list = re.finditer(regex, weights, re.MULTILINE | re.DOTALL)
        for num, w in enumerate(weight_list):
                for group in range(0, len(w.groups())):
                        split_list.append(w.group(1))

        selections_list = []

        for s in split_list:
                s = re.sub('\s+', ' ', s).strip()
                s = s.split(" ")
                inner_list = []
                for i in indices:
                        inner_list.append(float(s[i]))
                selections_list.append(inner_list)

        tweet_dict = {}

        for i in range(0, len(indices)):
                tweet_index = computeIndices(selections_list, i)
                subtweet_list = compileTweets(tweet_index, title, i)
                topic = computeSimilarity(subtweet_list, title)
                tweet_dict[topic] = [subtweet_list]
        return tweet_dict

def computeSimilarity(subtweet_list, title):

        scores = {}
        for t in title:
                topic = [t, subtweet_list[0][0]]
                tfidf_vectorizer = TfidfVectorizer()
                topic_matrix = tfidf_vectorizer.fit_transform(topic)
                cosine = cosine_similarity(topic_matrix[0:1], topic_matrix[1:2])
                scores[float(cosine)] = t
        t = sorted(scores.keys())
        return scores[t[-1]]


def computeIndices(selected_list, topic_num):

        top_score = {}
        for k, v in enumerate(selected_list):
                val = v[topic_num]
                if val == max(v):
                        top_score[k] = val

        sorted_scores = sorted(top_score.items(), key=operator.itemgetter(1), reverse=True)
        return sorted_scores

def compileTweets(tweet_index, title, i):

        tweet_list = []
        for t in tweet_index:
				index = t[0]
				tweet = docs[index].split('~+&$!sep779++')[0]
				date = docs[index].split('~+&$!sep779++')[1]
				id = docs[index].split('~+&$!sep779++')[2]
				tweet_list.append([tweet, date, id])

        return tweet_list

left_categories = open("results/topics.txt", "r") #the categories for a tweet
left_weights = open("results/W_indices_u.txt", "r") #their weights
left_selections = open("results/topics.txt", "r") #categories selected

left_indices, left_title = getIndices(left_categories, left_selections)

left = open("results/unsupervised_utility_tweets.csv", "w")
left_tweets = getTweets(left_indices, left_weights, left_title)

left = csv.writer(left)
left.writerow(["Topic", "Tweets", "Date", "Number", "Permalink"])
i = 0
for k, v in left_tweets.iteritems():
        left.writerow([k])
        i += 1
        for val in v:
			for l in val:
				left.writerow(["", l[0], l[1], str(i), l[2]])

