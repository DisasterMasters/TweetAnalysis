import xlrd
import random

text_list = []

book = xlrd.open_workbook("results/randomize.xlsx")
sheet = book.sheet_by_index(0)
for row in range(sheet.nrows):
	text = str(sheet.cell(row, 0))
	if "empty:u" not in text:
		text = text[7:-1]
		text_list.append(text)

file = open("results/random.txt", "w")
random.shuffle(text_list)
for t in text_list:
	file.write(t)
	file.write("\n")




