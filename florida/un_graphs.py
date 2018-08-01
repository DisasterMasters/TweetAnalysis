#pretty much the same as sup_graphs.py, but with slight differences

import csv 
import os
import io
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import datetime
import numpy as np
import sys
import mpld3
from mpld3 import plugins
from mpld3.utils import get_id

date_dict = {}
count_dict = {}

count = 0.0

name = sys.argv[1]

f = open("results/unsupervised_"+ name + "_tweets.csv", "r") #media, utility, gov, or nonprofit


label_titles = []

csv_f = csv.reader(f)
next(csv_f, None) #skip header
for row in csv_f:
	if row[0] != '':
		label_titles.append(row[0])
	elif row[0] == '':
		date = row[2]
		date = datetime.datetime.strptime(date, "%Y-%m-%d")
		if row[3] not in date_dict:
			date_dict[row[3]] = [date]
		else:
			date_dict[row[3]].append(date)
		if date not in count_dict:
			count_dict[date] = 1.0
		else:
			count_dict[date] += 1.0

fig, ax = plt.subplots()
colors = len(date_dict.keys())
cm = plt.get_cmap('tab20')
fig = plt.figure()
ax = fig.add_subplot(111)
ax.set_prop_cycle(color=[cm(1.*i/colors) for i in range(colors)])


ordered_titles = []

line_collections = []

for k, v in date_dict.iteritems():
	tmp_dict = {}
	for vals in v:
		if vals not in tmp_dict:
			tmp_dict[vals] = 0.0
		else:
			tmp_dict[vals] += 1.0
	if sys.argv[2] == 'p':
		for j, v in tmp_dict.iteritems():
			tmp_dict[j] /= count_dict[j]
	x, y = zip(*sorted(tmp_dict.items()))
	line_collections.append(ax.plot_date(x, y, '-'))
	ordered_titles.append(label_titles[int(k) - 1])

interactive_legend = plugins.InteractiveLegendPlugin(line_collections, ordered_titles)

ax.set_ylabel("Number of Tweets Per Day")
ax.set_xlabel("Time")
if sys.argv[2] == 'p':
	name += "_percentage"
elif sys.argv[2] == 'n':
	name += "_counts"
#update later
ax.set_title("Topics Distribution During Hurricane Irma (NMF) (" + name + ")")
mpld3.plugins.connect(fig, interactive_legend)
fig.set_size_inches(65.5, 15.5)
#fig.savefig("results/" + name + "_nmf_graph.png")
html_string = mpld3.fig_to_html(fig)
#mpld3.show()
figure = open('results/' + name + '_nmf.html', 'w')
figure.write(html_string)
