# Uncomment for Python 2
# from __future__ import *

import csv
import glob
import os
import re
import sys
from datetime import datetime
from randomforrest_filter import build_model
import pandas as pd
from tqdm import tqdm


def row_extractor(fname, row):
    if len(row) > 8:
        tweet = row[5]
        link = row[8]
        if fname[0] == '@' or fname == "FloridaMediaTweets.csv":
            tweet = row[4]
            link = row[9]

        if "Translated" in fname:
            try:
                tweet = row[9]
            except:
                pass

        # if any((txt in tweet) for txt in lex) and not (any(txt in tweet for txt in exclude)):
        # deal with messy dates
        date_text = row[1].split(' ')[0]
        date_text = date_text.strip()
        date = ''
        if '/' in date_text:
            date = datetime.strptime(date_text, '%m/%d/%Y')
        elif '-' in date_text:
            date = datetime.strptime(date_text, '%Y-%m-%d')
        '''
        prob dont need this
        #make sure date is in september
        start = datetime(year=2017, month=9, day=1)
        end = datetime(year=2017, month=9, day=30)
        '''
        # split tweet row to make sure there are more than 3 words in it
        tmp = tweet.split(' ')
        # if start < date < end and len(tmp) > 3:
        clean_text = tweet
        # strip extra whitespace
        clean_text = clean_text.strip()
        # remove http/s link
        clean_text = re.sub(r'https?([^\s]+)', '', clean_text)
        # remove __NEWLINE__
        clean_text = re.sub(r'__NEWLINE__', '', clean_text)
        # remove pic.twitter.com
        clean_text = re.sub(r'pic.twitter.com([^\s]+)', '', clean_text)
        # remove @handle
        clean_text = re.sub(r'@([^\s]+)', '', clean_text)
        # remove via @___
        clean_text = re.sub(r'via @([^\s]+)', '', clean_text)
        # put tweet, date, and permalink into unique separator for file
        # text = clean_text + '~+&$!sep779++' + date_text + '~+&$!sep779++' + link
        return clean_text, date_text, link


if len(sys.argv) < 3:
    print("Usage: " + sys.argv[0] +
          " [category or input_dir] [location or output_file]")
    exit(1)

# Get the root directory of the project
root_dir = os.path.abspath(__file__)
cur_dir = "."

while cur_dir != '':
    if os.path.isdir(root_dir + '/.git'):
        break

    root_dir, cur_dir = os.path.split(root_dir)

if cur_dir == '':
    print("Error: can't locate root directory")
    exit(1)

# enter argument as to which file the program will take to convert to test data
category = sys.argv[1].lower()
place = sys.argv[2].lower()
exclude = ['puerto rico', 'virgin islands', 'texas', 'houston', 'maria',
           'jose', 'harvey', 'katrina', 'florida']

# find directory containing files with data from tweet scraper
if place in {'puertorico', 'texas', 'florida'}:
    direc = root_dir + "/DATA/" + place + "_data/" + category + "_data"

    if place == 'puertorico':
        exclude.remove('puerto rico')
    elif place == 'texas':
        exclude.remove('texas')
        exclude.remove('houston')
    elif place == 'florida':
        exclude.remove('florida')
else:
    direc = sys.argv[1]

# add cmd line denoting location and change data directories based on that
# choose between utility, media, nonprofit, and government
if category in {'utility', 'media', 'nonprofit', 'gov', 'env'}:
    outfile = root_dir + '/' + place + '/training_data/' + category + '_data_(not_utility_source).txt'
else:
    outfile = sys.argv[2]

# find directory containing files with data from tweet scraper
# direc = "../DATA/florida_data/" + typeoffile + "_data"


df_dict = {"Tweet": [], "Date": [], "Link": []}

tfidf, clf = build_model()

for fname in tqdm(glob.glob(direc + "**/*", recursive=True)):
    if (os.path.isfile(fname)):
        # print(fname)
        try:
            f = open(fname, encoding="ISO-8859-1")
        except:
            pass
        name = str(fname)
        # open csv file if found
        if fname.endswith(".csv"):
            csv_f = csv.reader(f)
        # open txt file if found
        elif fname.endswith(".txt"):
            csv_f = csv.reader(f, delimiter='|')
        header = next(csv_f, None)  # skip header, save for output
        for row in csv_f:
            try:
                fname = os.path.basename(fname)
                tweet = row[5]
                link = row[8]
                if fname[0] == '@' or fname == "FloridaMediaTweets.csv":
                    tweet = row[4]
                    link = row[9]
                if row[0] not in link:
                    clean_text, date_text, link = row_extractor(fname, row)
                    df_dict["Tweet"].append(clean_text)
                    df_dict["Date"].append(date_text)
                    df_dict["Link"].append(link)

                    df = df.append({'Tweet': clean_text, 'Date': date_text, 'Link': link}, ignore_index=True)
            except:
                pass

df = pd.DataFrame.from_dict(df_dict)
x = tfidf.transform(df['Tweet'])
df['rfresults'] = clf.predict(x)
df = df[df.rfresults != 0]
df = df.drop('rfresults', 1)
# open output file
df.to_csv(outfile)
