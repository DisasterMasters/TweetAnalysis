import csv 
import os
import io
from crisislex import Crisislex
import pandas as pd
from datetime import datetime
import sys

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
			if flag == 0:
				output.writerow(header)
				flag = 1
			for row in csv_f:
				if len(row) > 9:
					if any(txt in row[4] for txt in lex):
						date = row[1][:-5]
						date = date.strip()
						if '/' in date:
							date = datetime.strptime(date, '%m/%d/%Y')
						elif '-' in date:
							date = datetime.strptime(date, '%Y-%m-%d')
						start = datetime(year=2017, month=9, day=1)
						end = datetime(year=2017, month=9, day=30)
						if start < date < end:	
							text = row[4] + '~+&$!sep779++' + row[1][:-5] + '~+&$!sep779++' + row[9]
							text = text + "\t"
							text = text.decode('utf-8', errors='ignore')
							file.write(text)
							#output.writerow(row)

file.close()
				

