import csv 
import os
import io
from crisislex import Crisislex
import pandas as pd
from datetime import datetime
import sys
import re

#typeoffile = "nonprofit"
#outfile = "training_data/n_dates.txt"

typeoffile = sys.argv[1]
outfile = 'training_data/dates.txt'
if typeoffile == 'media':
	outfile = 'training_data/m_dates.txt'


file = io.open(outfile, "w", encoding="utf-8", errors="ignore")
#csv_f= open("results/media_lexfiltered.csv", "w")
#output = csv.writer(csv_f)

c_lex = Crisislex()
lex = c_lex.lex

flag = 0

for dirs, subdirs, files in os.walk(typeoffile):
	for fname in files:
		if fname.endswith(".csv"): 
			f = open(dirs + "/" + fname, "r")
			name = str(fname)
			csv_f = csv.reader(f)
			if name != 'FloridaMediaTweets.csv' and typeoffile == 'media':
				csv_f = csv.reader(f, delimiter='|')
			header = next(csv_f, None) #skip header, save for output
			#if flag == 0:
				#output.writerow(header)
				#flag = 1
			for row in csv_f:
				if len(row) >= 8:
					if any(txt in row[4] for txt in lex):
						date_text = row[1].split(' ')[0]
						date_text = date_text.strip()
						date = ''
						if '/' in date_text:
							date = datetime.strptime(date_text, '%m/%d/%Y')
						elif '-' in date_text:
							date = datetime.strptime(date_text, '%Y-%m-%d')
						start = datetime(year=2017, month=9, day=1)
						end = datetime(year=2017, month=9, day=30)
						tmp = row[4].split(' ')
						if start < date < end and len(tmp) > 3: 
							clean_text = row[4]
							#remove url, via @_, and @handles
							clean_text = re.sub(r'https?([^\s]+)', '', clean_text)
							clean_text = re.sub(r'pic.twitter.com([^\s]+)', '', clean_text)
							clean_text = re.sub(r'@([^\s]+)', '', clean_text)
							clean_text = re.sub(r'via @([^\s]+)', '', clean_text)
							text = clean_text + '~+&$!sep779++' + date_text + '~+&$!sep779++' + row[9] #need to make condition here
							text = text + "\t"
							text = text.decode('utf-8', errors='ignore')
							file.write(text)
							#output.writerow(row)

file.close()
				

