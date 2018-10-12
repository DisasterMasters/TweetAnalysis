from urllib.request import urlopen
import csv
import os
import tempfile
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

regex = re.compile(r"pic\.twitter\.com/\S+")

magic_numbers = {
    0x424D,                     # Windows Bitmap
    0xFFD8FF,                   # JPEG
    0x0000000C6A5020200D0A870A, # JPEG 2000
    0x89504e470d0a1a0a,         # Portable Network Graphics
    0x5031,
    0x5032,
    0x5033,
    0x5034,
    0x5035,
    0x5036,                     # Netpbm
    0x59A66A95,                 # Sun Raster
    0x4d4d002a,
    0x49492a00                  # Tagged Image File Format
}

class ScrapyDialect(csv.Dialect):
    delimiter = "|"
    quotechar = "'"
    doublequote = True
    skipinitialspace = False
    lineterminator = "\n"
    quoting = csv.QUOTE_MINIMAL

for row in csv.DictReader(sys.stdin, dialect = ScrapyDialect, quoting = csv.QUOTE_ALL):
    text = row["text"]
    url_match = regex.search(text)

    if url_match is None:
        continue

    try:
        tweet_id = int(urlopen("https://" + text[url_match.start():url_match.end()]).geturl().split("/")[5])
    except Exception:
        continue

    tweet = twitter.get_status(tweet_id,
        include_entities = True,
        tweet_mode = "extended",
        monitor_rate_limit = True,
        wait_on_rate_limit = True
    )

    media = [t["media_url"] for t in tweet.entities["media"]]

    for m in media:
        try:
            slurped = urlopen(m).read()
        except Exception:
            continue

        if not any((int.from_bytes(slurped[0:n], byteorder = 'big') in magic_numbers) for n in range(2, 13)):
            continue

        with tempfile.NamedTemporaryFile(delete = False) as fd:
            save_name = fd.name
            fd.write(slurped)

        img = cv2.imread(save_name)
        os.remove(save_name)

        # Do something with img here
