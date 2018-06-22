import csv
import re


file = open("clean_up.csv", "r")
c_file = csv.reader(file)
header = c_file.next()
outfile = open("cleaned.csv", "w")
out_w = csv.writer(outfile)
out_w.writerow(header)
for row in c_file:
	tweet = row[12]
	#check if just a picture
	#pic_check = re.match(r'^pic\.twitter\.com\/.*$', tweet)
	#check if three or less words (seems to cover pictures as well)
	word_check = tweet.split(' ')
	if len(word_check) > 3:
		out_w.writerow(row)
