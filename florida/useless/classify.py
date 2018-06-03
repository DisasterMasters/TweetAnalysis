import csv


tweet_dict = {}

with open('nmf_cosine_results.csv', 'r') as csvt:
	csv_reader = csv.reader(csvt)
	next(csv_reader) #skip first row  of csv file
	i = 0
	tweet_list = []
	for row in csv_reader:
		if row[0] != '':
			i += 1
			tweet_list = []
		if row[1] != '':
			tweet_list.append(row[1])
		tweet_dict[i] = tweet_list


output = open("topic_assignments.csv", "w")
cwriter = csv.writer(output)
cwriter.writerow(['A Before', 'B Before', 'A After', 'B After', 'Topic Num', 'Tweet'])


with open('topic_classifications.csv', 'r') as csvt:
	csv_reader = csv.reader(csvt)
	next(csv_reader) #skip first row  of csv file
	next(csv_reader) #skip first row  of csv file
	for row in csv_reader:
		tweet = str(row[5])
		a_before = row[0]
		b_before = row[1]
		a_after = row[3]
		b_after = row[4]
		for k, list in tweet_dict.iteritems():
			for v in list:
				if tweet == v:
					print tweet, v
					cwriter.writerow([a_before, b_before, a_after, b_after, k, tweet])

