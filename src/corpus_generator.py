import csv
import os
import re
import sys
from datetime import datetime

import pandas as pd
from tqdm import tqdm

csv.field_size_limit(sys.maxsize)

import glob

datafolder = r"/home/manny/PycharmProjects/TweetAnalysis/DATA/"
print("Loading Corpus")

df_dict = {"Tweet": [], "Date": [], "Link": []}
for fname in tqdm(glob.glob(datafolder + "**/*", recursive=True)):
    if (os.path.isfile(fname)):
        print(fname)
        try:
            f = open(fname)
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
        try:
            for row in csv_f:
                # skip empty row or row that is too short
                if len(row) > 8:
                    # make sure tweet has word from crisislex
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
                        # print(tweet)

                    if "caribbean equipment" in tweet:
                        print(fname, "FOUND")
                        input(" WHAY DO D")


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
                    text = clean_text + '~+&$!sep779++' + date_text + '~+&$!sep779++' + link

                    df_dict["Tweet"].append(tweet)
                    df_dict["Date"].append(date_text)
                    df_dict["Link"].append(link)

                    # df = df.append({'Tweet': clean_text, 'Date': date_text, 'Link': link}, ignore_index=True)
                    # tab separate each tweet line
                    text = text + "\t"
                    # print(text)
        except:
            pass


df = pd.DataFrame.from_dict(df_dict)
df.to_csv('corpus.csv')
