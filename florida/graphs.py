import csv 
import os
import io
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import datetime
import numpy as np

date_dict = {}
f = open("results/representation.csv", "r")
csv_f = csv.reader(f)
next(csv_f, None) #skip header
for row in csv_f:
	if row[0] == '':
		date_dict[row[3]] = []
f.close()

count = 0.0
f = open("results/representation.csv", "r")
csv_f = csv.reader(f)
next(csv_f, None) #skip header
for row in csv_f:
	if row[0] == '':
		date = row[2].strip()
		date = date.replace('+', '')
		date = datetime.datetime.strptime(date, "%Y-%m-%d")
		date_dict[row[3]].append(date)
		count += 1.0

file = open("len_ut.txt", "r")
num = int(file.read())


colors = len(date_dict.keys())
cm = plt.get_cmap('gist_rainbow')
fig = plt.figure()
ax = fig.add_subplot(111)
ax.set_prop_cycle(color=[cm(1.*i/colors) for i in range(colors)])


i = 1
for k, v in date_dict.iteritems():
	tmp_dict = {}
	for vals in v:
		if vals not in tmp_dict:
			tmp_dict[vals] = 0.0
		else:
			tmp_dict[vals] += 1.0
	for k, v in tmp_dict.iteritems():
		tmp_dict[k] /= num
	x, y = zip(*sorted(tmp_dict.items()))
	plt.plot_date(x, y, '-', label=str(i))
	plt.legend()
	i += 1	

plt.ylabel("Number of Tweets Per Day")
plt.xlabel("Time")
plt.title("Topics Distribution During Hurricane Irma (Unsupervised)")
fig = plt.gcf()
fig.set_size_inches(20.5, 12.5)
fig.savefig('unsupervised_graph.png')
