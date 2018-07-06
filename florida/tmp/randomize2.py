import csv
import random

text_list = []

file = open("results/representation.csv", "r")
csv_read = csv.reader(file)
next(csv_read)
for row in csv_read:
	if row[0] == '':
		text = str(row[1])
		id = str(row[4])
		text_list.append([text, id])

file = open("results/random_utility_latest.csv", "w")
csv_write = csv.writer(file)
random.shuffle(text_list)
for t in text_list:
	csv_write.writerow([t[0], t[1]])




