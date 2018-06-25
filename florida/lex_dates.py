import csv 
import os
import io
from crisislex import Crisislex
import pandas as pd

file = io.open("training_data/m_dates.txt", "w", encoding="utf-8", errors="ignore")
csv_f= open("results/media_lexfiltered.csv", "w")
output = csv.writer(csv_f)

c_lex = Crisislex()
lex = c_lex.lex

flag = 0

for dirs, subdirs, files in os.walk("media"):
	for fname in files:
		if fname.endswith(".csv"): 
			f = open(dirs + "/" + fname, "r")
			name = str(fname)
			csv_f = csv.reader(f)
			header = next(csv_f, None) #skip header, save for output
			if flag == 0:
				output.writerow(header)
				flag = 1
			for row in csv_f:
				if len(row) > 9:
					if any(txt in row[4] for txt in lex):
						text = row[4] + '~+&$!sep779++' + row[1][:-5] + '~+&$!sep779++' + row[9]
						text = text + "\t"
						text = text.decode('utf-8', errors='ignore')
						file.write(text)
						output.writerow(row)

file.close()
				

