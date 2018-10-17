import csv
import math
import re
import sys
import unicodedata

import tweepy

def tweet_to_dict(tweet):
    def get_info():
        cos_phi = math.sqrt(math.pi / 2)
        sec_phi = math.sqrt(2 / math.pi)
        R_earth = 6371

        # Map polygon coordinates to points on an equal-area cylindrical projection (Smyth equal-surface)
        polygons = [[(math.radians(lon) * cos_phi * R_earth, math.sin(math.radians(lat)) * sec_phi * R_earth)
                     for [lat, lon] in p]
                     for p in tweet.place.bounding_box.coordinates]

        areas = []
        comxs = []
        comys = []

        for p in polygons:
            segments = list(zip(p, p[1:] + p[:1]))

            a = sum(x0 * y1 - x1 * y0 for (x0, y0), (x1, y1) in segments) * 0.5

            cx = sum((x0 + x1) * (x0 * y1 - x1 * y0) for (x0, y0), (x1, y1) in segments) / (6 * a)
            cy = sum((y0 + y1) * (x0 * y1 - x1 * y0) for (x0, y0), (x1, y1) in segments) / (6 * a)

            areas.append(abs(a))
            comxs.append(cx)
            comys.append(cy)

        total_area = sum(areas)

        cx = sum(c * a for c, a in zip(comxs, areas)) / total_area
        cy = sum(c * a for c, a in zip(comys, areas)) / total_area

        # Unmap from projection to exact coordinates
        lat = math.degrees(math.asin(cy * (cos_phi / R_earth)))
        lon = math.degrees(cx * (sec_phi / R_earth))

        r = math.sqrt(total_area / math.pi)

        return (lat, lon, r)

    tweet_dict = {
        "id": tweet.id,
        #"text": tweet.full_text,
        "userid": tweet.author.id,
        "username": tweet.author.screen_name,
        "reply_to": tweet.in_reply_to_status_id,
        "date": tweet.created_at,
        "retweets": tweet.retweet_count,
        "favorites": tweet.favorite_count,
        "hashtags": ["#" + e["text"] for e in tweet.entities["hashtags"]] +
                    ["$" + e["text"] for e in tweet.entities["symbols"]],
        "mentions": [e["id"] for e in tweet.entities["user_mentions"]],
        "urls": [e["expanded_url"] for e in tweet.entities["urls"]],
        "lang": None if tweet.lang == "und" else tweet.lang,
        "permalink": "https://twitter.com/statuses/" + tweet.id_str
    }

    try:
        tweet_dict["text"] = tweet.full_text
    except AttributeError:
        tweet_dict["text"] = tweet.text

    if "media" in tweet.entities:
        tweet_dict["media"] = [e["media_url"] for e in tweet.entities["media"]]
    else:
        tweet_dict["media"] = []

    if tweet.coordinates is not None and tweet.coordinates["type"] == "Point":
        [tweet_dict["coord_lat"], tweet_dict["coord_lon"]] = tweet.coordinates["coordinates"]
        tweet_dict["coord_err"] = 0.0
    elif tweet.place is not None:
        tweet_dict["coord_lat"], tweet_dict["coord_lon"], tweet_dict["coord_err"] = get_info()
    else:
        tweet_dict["coord_lat"] = None
        tweet_dict["coord_lon"] = None
        tweet_dict["coord_err"] = None

    # Strip whitespace and normalize
    clean_text = unicodedata.normalize('NFC', tweet_dict["text"].strip())
    # Remove characters outside BMP (emojis)
    #clean_text = "".join(c for c in clean_text if ord(c) <= 0xFFFF)
    # Remove newlines and tabs
    clean_text = clean_text.replace("\\n", " ").replace("\t", " ")
    # Remove HTTP(S) link
    clean_text = re.sub(r"https?://\S+", "", clean_text)
    # Remove pic.twitter.com
    clean_text = re.sub(r"pic.twitter.com/\S+", "", clean_text)
    # Remove @handle at the start of the tweet
    clean_text = re.sub(r"\A(@\w+ ?)*", "", clean_text)
    # Remove via @handle
    clean_text = re.sub(r"via @\w+", "", clean_text)

    tweet_dict["cleantext"] = clean_text

    # Convert everything to a text field
    for k in tweet_dict.keys():
        v = tweet_dict[k]

        if type(v) is not str:
            if v is None:
                tweet_dict[k] = ""
            elif type(v) is list:
                tweet_dict[k] = ",".join(str(i) for i in v)
            else:
                tweet_dict[k] = str(v)

        tweet_dict[k] = tweet_dict[k].replace("\n", "__NEWLINE__")
        tweet_dict[k] = tweet_dict[k].replace("|", "__PIPE__")

    return tweet_dict

class ScrapyDialect(csv.Dialect):
    delimiter = "|"
    quotechar = "'"
    doublequote = True
    skipinitialspace = False
    lineterminator = "\n"
    quoting = csv.QUOTE_MINIMAL

class CsvListener(tweepy.StreamListener):
    def __init__(self, fd):
        super().__init__()

        self.csv = csv.DictWriter(fd, dialect = ScrapyDialect, fieldnames = [
            "id",
            "text",
            "cleantext",
            "userid",
            "username",
            "reply_to",
            "date",
            "retweets",
            "favorites",
            "hashtags",
            "mentions",
            "media",
            "urls",
            "lang",
            "coord_lat",
            "coord_lon",
            "coord_err",
            "permalink"
        ])

        self.csv.writeheader()

    def on_status(self, status):
        if hasattr(status, "extended_tweet"):
            for k, v in status.extended_tweet.items():
                setattr(status, k, v)

        self.csv.writerow(tweet_to_dict(status))

    def on_error(self, status_code):
        if status_code == 420:
            return False

if __name__ == "__main__":
    # These are unique for this program
    __auth = tweepy.OAuthHandler(
        "ZFVyefAyg58PTdG7m8Mpe7cze",
        "KyWRZ9QkiC2MiscQ7aGpl5K2lbcR3pHYFTs7SCVIyxMlVfGjw0"
    )
    __auth.set_access_token(
        "1041697847538638848-8J81uZBO1tMPvGHYXeVSngKuUz7Cyh",
        "jGNOVDxllHhO57EaN2FVejiR7crpENStbZ7bHqwv2tYDU"
    )

    strm = tweepy.Stream(
        auth = __auth,
        listener = CsvListener(sys.stdout),
        tweet_mode = "extended",
        monitor_rate_limit = True,
        wait_on_rate_limit = True
    )
    del __auth

    strm.filter(track = ["Hurricane Florence", "#Florence"])
