from urllib.request import urlopen
import csv
import os
import itertools
import sys
import re

import cv2
import numpy as np
import tweepy


__auth = tweepy.OAuthHandler(
    "ZFVyefAyg58PTdG7m8Mpe7cze",
    "KyWRZ9QkiC2MiscQ7aGpl5K2lbcR3pHYFTs7SCVIyxMlVfGjw0"
)
__auth.set_access_token(
    "1041697847538638848-8J81uZBO1tMPvGHYXeVSngKuUz7Cyh",
    "jGNOVDxllHhO57EaN2FVejiR7crpENStbZ7bHqwv2tYDU"
)

twitter = tweepy.API(__auth)

magic_numbers = {
    0x424D,                     # Windows Bitmap
    0xFFD8FF,                   # JPEG
    0x0000000C6A5020200D0A870A, # JPEG 2000
    0x89504e470d0a1a0a,         # Portable Network Graphics
    0x5031,                     # Netpbm
    0x5032,
    0x5033,
    0x5034,
    0x5035,
    0x5036,
    0x59A66A95,                 # Sun Raster
    0x4D4D002A,                 # Tagged Image File Format
    0x49492A00
}

class ScrapyDialect(csv.Dialect):
    delimiter = "|"
    quotechar = "'"
    doublequote = True
    skipinitialspace = False
    lineterminator = "\n"
    quoting = csv.QUOTE_MINIMAL


orb = cv2.ORB_create()
tweets = []

def get_tids(fin):
    regex = re.compile(r"pic\.twitter\.com/\S+")
    ret = []

    for row in csv.DictReader(fin, dialect = ScrapyDialect, quoting = csv.QUOTE_ALL):
        text = row["text"]
        url_match = regex.search(text)

        if url_match is not None:
            try:
                tweet_id = int(urlopen("https://" + text[url_match.start():url_match.end()]).geturl().split("/")[5])
            except Exception:
                continue

            ret.append(tweet_id)

    return ret

def get_imgs(tid):
    ret = []

    tweet = twitter.get_status(tweet_id,
        include_entities = True,
        tweet_mode = "extended",
        monitor_rate_limit = True,
        wait_on_rate_limit = True
    )

    for e in tweet.entities["media"]:
        url = e["media_url"]

        try:
            img_data = urlopen(url).read()
        except Exception:
            continue

        # OpenCV uses a C assertion (which the Python interpreter can't recover
        # from) to make sure the file type is valid, so check the file type
        # ahead of time
        if not any((int.from_bytes(img_data[0:n], byteorder = 'big') in magic_numbers) for n in range(2, 13)):
            continue

        img = cv2.imdecode(np.fromstring(img_data, dtype = np.uint8), cv2.IMREAD_GRAYSCALE)
        ret.append(img)

    return ret

# Don't run this yet
exit(0)

matcher = cv2.BFMatcher()

for (a_tweet, a_img, a_kp, a_des), (b_tweet, b_img, b_kp, b_des) in itertools.product(tweets, tweets):
    if a_tweet.id == b.tweet.id:
        continue

    matches = matcher.knnMatch(a_des, b_des, k = 2)
    matches = [(m, n) for m, n in matches if m.distance < 0.75 * n.distance]

    if len(matches) >= 25:
        # Do something with a_tweet and b_tweet
        pass
