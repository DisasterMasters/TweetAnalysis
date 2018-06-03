import csv 
import os
import io
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import datetime
import numpy as np
from difflib import SequenceMatcher

date_dict = {}
f = open("training_data/random_100s_F.csv", "r")
csv_f = csv.reader(f)
next(csv_f, None) #skip header
for row in csv_f:
	date_dict[row[0].split('.')[0]] = []
f.close()

def lookUpDate(tweet):
	
	match_dict = {}
	for dirs, subdirs, files in os.walk("utility"):
		for fname in files:
			if fname.endswith(".csv"): 
				f = open(dirs + "/" + fname, "r")
				name = str(fname)
				csv_f = csv.reader(f)
				next(csv_f, None) #skip header
				for row in csv_f:
					match = SequenceMatcher(None, tweet, row[4]).ratio()
					if match > 0.4:
						match_dict[match] = [tweet, row[4], row[1][:-5]]
	sorted_m = sorted(match_dict.keys())
	print match_dict[sorted_m[-1]]
					#	r	eturn row[1][:-5]


f = open("training_data/random_100s_F.csv", "r")
csv_f = csv.reader(f)
next(csv_f, None) #skip header
for row in csv_f:
#	date = row[2].strip()
#	date = date.replace('+', '')
	date = lookUpDate(row[1])
	date = datetime.datetime.strptime(date, "%Y-%m-%d")
	date_dict[row[0].split('.')[0]].append(date)


colors = len(date_dict.keys())
cm = plt.get_cmap('gist_rainbow')
fig = plt.figure()
ax = fig.add_subplot(111)
ax.set_prop_cycle(color=[cm(1.*i/colors) for i in range(colors)])

for k, v in date_dict.iteritems():
	tmp_dict = {}
	for vals in v:
		if vals not in tmp_dict:
			tmp_dict[vals] = 0
		else:
			tmp_dict[vals] += 1
	x, y = zip(*sorted(tmp_dict.items()))
	plt.plot_date(x, y, '-', label=str(k))
	plt.legend()

plt.ylabel("Number of Tweets Per Day")
plt.xlabel("Time")
plt.title("Topics Distribution During Hurricane Irma (Supervised)")
fig = plt.gcf()
fig.set_size_inches(20.5, 12.5)
fig.savefig('supervised_graph.png')
