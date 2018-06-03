import csv 
import os
import io

file = io.open("utility/tweets.txt", "w", encoding="utf-8", errors="ignore")

for dirs, subdirs, files in os.walk("data"):
	for fname in files:
		if fname.endswith(".csv"): 
			f = open(dirs + "/" + fname, "r")
			name = str(fname)
			csv_f = csv.reader(f)
			next(csv_f, None) #skip header
			for row in csv_f:
				text = row[4]
				text = text + "\t"
				text = text.decode('utf-8', errors='ignore')
				file.write(text)

file.close()
				

