import csv 
import os
import io
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import datetime
import numpy as np
import sys

date_dict = {}
count_dict = {}

count = 0.0

name = sys.argv[1]

f = open("results/" + name + "_supervised.csv", "r")

csv_f = csv.reader(f)
next(csv_f, None) #skip header
for row in csv_f:
	date = row[2]
	date = datetime.datetime.strptime(date, "%m/%d/%Y")
	if row[1] not in date_dict:
		date_dict[row[1]] = [date]
	else:
		date_dict[row[1]].append(date)
	if date not in count_dict:
		count_dict[date] = 1.0
	else:
		count_dict[date] += 1.0


colors = len(date_dict.keys())
cm = plt.get_cmap('gist_rainbow')
fig = plt.figure()
ax = fig.add_subplot(111)
ax.set_prop_cycle(color=[cm(1.*i/colors) for i in range(colors)])


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
	plt.plot_date(x, y, '-', label=str(k))
	plt.legend()

plt.ylabel("Number of Tweets Per Day")
plt.xlabel("Time")
if sys.argv[2] == 'p':
	name += "_percentage"
elif sys.argv[2] == 'n':
	name += "_counts"
plt.title("Topics Distribution During Hurricane Irma (Supervised) (" + name + ")")
fig = plt.gcf()
fig.set_size_inches(20.5, 12.5)
fig.savefig("results/" + name + "_supervised_graph.png")
