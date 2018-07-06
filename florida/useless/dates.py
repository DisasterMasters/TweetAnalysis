import csv 
import os
import io

file = io.open("training_data/m_dates.txt", "w", encoding="utf-8", errors="ignore")

for dirs, subdirs, files in os.walk("media"):
	for fname in files:
		if fname.endswith(".csv"): 
			f = open(dirs + "/" + fname, "r")
			name = str(fname)
			csv_f = csv.reader(f)
			next(csv_f, None) #skip header
			for row in csv_f:
				if len(row) > 9:
					text = row[4] + '~+&$!sep779++' + row[1][:-5] + '~+&$!sep779++' + row[9]
					text = text + "\t"
					text = text.decode('utf-8', errors='ignore')
					file.write(text)

file.close()
				

