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


def relevant_filter():
    from randomforrest_filter import build_model

    tfidf, clf = build_model()
    df = corpus2df()
    x = tfidf.transform(df['Tweet'])
    df['rfresults'] = clf.predict(x)
    df = df[df.rfresults != 0]
    df = df.drop('rfresults', 1)
    return df


def corpus2df():
    print("Loading Corpus")

    df_dict = {"Tweet": [], "Raw": [], "Date": [], "Link": [], "Category": []}
    for fname in tqdm(glob.glob(datafolder + "**/*", recursive=True)):
        if (os.path.isfile(fname) and (fname.endswith(".csv") or fname.endswith(".txt")) and "WCOORDS" not in fname):
            print(fname)

            category = ""

            if "media" in fname.lower():
                category = "Media"

            if "gov" in fname.lower():
                category = "Government"

            if "nonprofit" in fname.lower():
                category = "Nonprofit"

            if "utility" in fname.lower():
                category = "Utility"

            try:
                f = open(fname, encoding="ISO-8859-1")
            except:
                pass
            name = str(fname)
            # open csv file if found
            csv_f = csv.reader(f)

            if fname.endswith(".csv"):
                csv_f = csv.reader(f)
            # open txt file if found
            elif fname.endswith(".txt"):
                csv_f = csv.reader(f, delimiter='|')
            header = next(csv_f, None)  # skip header, save for output
            for row in csv_f:
                try:
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
                        rawtext = tweet
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
                        df_dict["Raw"].append(rawtext)
                        df_dict["Date"].append(date_text)
                        df_dict["Link"].append(link)
                        df_dict["Category"].append(category)

                        # df = df.append({'Tweet': clean_text, 'Date': date_text, 'Link': link}, ignore_index=True)
                        # tab separate each tweet line
                        text = text + "\t"
                        # print(text)
                except:
                    pass

    df = pd.DataFrame.from_dict(df_dict)
    return df


def corpus2csv():
    df = corpus2df()
    df.to_csv('corpus.csv')
    return


