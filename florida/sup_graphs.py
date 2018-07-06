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

f = open("results/" + name + "_supervised_rf.csv", "r")

csv_f = csv.reader(f)
next(csv_f, None) #skip header
for row in csv_f:
	if row[1] != '0':
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


fig, ax = plt.subplots()
colors = len(date_dict.keys())
cm = plt.get_cmap('tab20')
fig = plt.figure()
ax = fig.add_subplot(111)
ax.set_prop_cycle(color=[cm(1.*i/colors) for i in range(colors)])


label_titles = ['Unrelated', 'Preparedness', 'Response', 'Status', 'Impact', 'Looting', 'Price Gouging', 'Other indirect Impact', 'Recovery', 'Requesting Help', 'Relief Efforts', 'Commenting Government', 'Commenting Utility', 'Commenting Media Coverage', 'Thanks', 'Well Wishes', 'Related to Other Hurricanes']

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
	ordered_titles.append(label_titles[int(k)])

interactive_legend = plugins.InteractiveLegendPlugin(line_collections, ordered_titles)

ax.set_ylabel("Number of Tweets Per Day")
ax.set_xlabel("Time")
if sys.argv[2] == 'p':
	name += "_percentage"
elif sys.argv[2] == 'n':
	name += "_counts"
#update later
ax.set_title("Topics Distribution During Hurricane Irma (Random Forest) (" + name + ")")
mpld3.plugins.connect(fig, interactive_legend)
fig.set_size_inches(26.5, 12.5)
fig.savefig("results/" + name + "_rf_graph.png")
html_string = mpld3.fig_to_html(fig)
#mpld3.show()
figure = open('results/' + name + '.html', 'w')
figure.write(html_string)
