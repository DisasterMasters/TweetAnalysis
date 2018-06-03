import csv
import random

text_list = []

file = open("results/one100more.csv", "r")
csv_read = csv.reader(file)
next(csv_read)
for row in csv_read:
	text = str(row[0])
	text_list.append(text)

file = open("results/random_100s.csv", "w")
csv_write = csv.writer(file)
random.shuffle(text_list)
for t in text_list:
	csv_write.writerow([t])




