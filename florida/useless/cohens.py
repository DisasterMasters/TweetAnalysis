import csv
import numpy as np

def column(matrix, i):
	return [row[i] for row in matrix]


def compute_kappa(matrix):
	row_total = []
	column_total = []
	agreements = 0

	for i in range(0, 12):
		row_total.append(np.sum(matrix[i]))
		column_total.append(np.sum(column(matrix, i)))
		agreements = agreements + matrix[i][i]

	exp_freq = []
	overall_total = np.sum(row_total)

	for i in range(0, 12):	
		ef = (float(row_total[i]) * float(column_total[i])) / float(overall_total)
		exp_freq.append(ef)

	sum_ef = np.sum(exp_freq)

	kappa = (agreements - sum_ef)/(overall_total - sum_ef)
	return kappa


w, h = 12, 12
matrix_before = [[0 for x in range(w)] for y in range(h)]
matrix_after = [[0 for x in range(w)] for y in range(h)]


with open('topic_classifications.csv', 'r') as csvt:
	csv_reader = csv.reader(csvt)
	next(csv_reader) #skipping first two rows of csv file
	next(csv_reader)
	for row in csv_reader:
		a_b = int(row[0]) - 1
		b_b = int(row[1]) - 1
		a_a = int(row[3]) - 1
		b_a = int(row[4]) - 1
		matrix_before[a_b][b_b] += 1
		matrix_after[a_a][b_a] += 1


kappa_before = compute_kappa(matrix_before)
kappa_after = compute_kappa(matrix_after)

print "Kappa before consensus:", kappa_before
print "Kappa after consensus:", kappa_after
