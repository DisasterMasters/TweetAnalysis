import csv 
import os
import io
from crisislex import Crisislex
import pandas as pd
from datetime import datetime
import sys
import re

#enter argument as to which file the program will take to convert to test data
typeoffile = sys.argv[1]
exclude = []

#choose between utility, media, nonprofit, and government
if typeoffile == 'utility':
	outfile = 'training_data/utility_data.txt'
if typeoffile == 'media':
	outfile = 'training_data/media_data.txt'
	exclude = ['puerto rico', 'virgin islands', 'texas', 'houston', 'maria', 'jose', 'harvey', 'katrina']
if typeoffile == 'nonprofit':
	outfile = 'training_data/nonprofit_data.txt'
if typeoffile == 'gov':
	outfile = 'training_data/gov_data.txt'

#find directory containing files with data from tweet scraper
direc = "../DATA/florida_data/" + typeoffile + "_data"

#open output file
file = io.open(outfile, "w", encoding="utf-8")

#import crisislex to filter out tweets without crisislex keywords in them
c_lex = Crisislex()
lex = c_lex.lex

flag = 0

#walk through files
for dirs, subdirs, files in os.walk(direc):
	for fname in files:
		f = open(dirs + "/" + fname, "r")
		name = str(fname)
		#open csv file if found
		if fname.endswith(".csv"):
			csv_f = csv.reader(f)
		#open txt file if found
		elif fname.endswith(".txt"):
			csv_f = csv.reader(f, delimiter = '|')
		header = next(csv_f, None) #skip header, save for output
		for row in csv_f:
			#skip empty row or row that is too short
			if len(row) > 8: 
				#make sure tweet has word from crisislex
				tweet = row[5]
				link = row[8]
				if fname[0] == '@' or fname == "FloridaMediaTweets.csv":
					tweet = row[4]
					link = row[9]
				if any(txt in tweet for txt in lex) and not(any(txt in tweet for txt in exclude)):
					#deal with messy dates
					date_text = row[1].split(' ')[0]
					date_text = date_text.strip()
					date = ''
					if '/' in date_text:
						date = datetime.strptime(date_text, '%m/%d/%Y')
					elif '-' in date_text:
						date = datetime.strptime(date_text, '%Y-%m-%d')
					#make sure date is in september
					start = datetime(year=2017, month=9, day=1)
					end = datetime(year=2017, month=9, day=30)
					#split tweet row to make sure there are more than 3 words in it
					tmp = tweet.split(' ')
					if start < date < end and len(tmp) > 3:
						clean_text = tweet 
						#strip extra whitespace
						clean_text = clean_text.strip()
						#remove http/s link
						clean_text = re.sub(r'https?([^\s]+)', '', clean_text)
						#remove __NEWLINE__
						clean_text = re.sub(r'__NEWLINE__', '', clean_text)
						#remove pic.twitter.com
						clean_text = re.sub(r'pic.twitter.com([^\s]+)', '', clean_text)
						#remove @handle
						clean_text = re.sub(r'@([^\s]+)', '', clean_text)
						#remove via @___
						clean_text = re.sub(r'via @([^\s]+)', '', clean_text)
						#put tweet, date, and permalink into unique separator for file
						text = clean_text + '~+&$!sep779++' + date_text + '~+&$!sep779++' + link
						#tab separate each tweet line
						text = text + "\t"
						#decode text 
						text = text.decode('utf-8', errors='ignore')
						file.write(text)

			

