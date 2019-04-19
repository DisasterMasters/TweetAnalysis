import collections
import csv
import math
import sys

from common import *
from marshalling import *

def get_extended_tags(r):
    def order(n):
        return str(-1 if n <= 0 else math.floor(math.log10(n)))

    s = {
        "favorite_order_" + order(r["favorite_count"]),
        "retweet_order_" + order(r["retweet_count"])
    }

    try:
        s.add("reply_order_" + order(r["reply_count"]))
    except KeyError:
        pass

    s |= {"#" + hashtag["text"] for hashtag in r["entities"]["hashtags"]}
    s |= {"@" + user_mention["screen_name"] for user_mention in r["entities"]["user_mentions"]}
    s |= {"$" + symbol["text"] for symbol in r["entities"]["symbols"]}

    try:
        s |= {"#" + hashtag["text"] for hashtag in r["extended_tweet"]["entities"]["hashtags"]}
    except KeyError:
        pass

    try:
        s |= {"@" + user_mention["screen_name"] for user_mention in r["extended_tweet"]["entities"]["user_mentions"]}
    except KeyError:
        pass

    try:
        s |= {"$" + symbol["text"] for symbol in r["extended_tweet"]["entities"]["symbols"]}
    except KeyError:
        pass

    return iter(s)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: " + sys.argv[0] + " <Pickled data>", file = sys.stderr)
        exit(-1)

    with opentunnel(), opendb() as db:
        rs = []

        for filename in sys.argv[1:]:
            ext = filename[filename.rfind("."):]

            cls = {
                ".csv": UnixCSVUnmarshaller,
                ".pkl": PickleUnmarshaller,
                ".bson": BSONUnmarshaller
            }[ext]

            with cls(filename) as unmarshaller:
                rs.extend(unmarshaller)

        with opencoll(db, "LabeledStatuses_Power_A", cleanup = False) as coll:
            rs.extend(coll.find())

        print("Calculating extended tags for tweets...")

        id_set = set()

        def map_f(r):
            id = r["retweeted_status"]["id"] if "retweeted_status" in r else r["id"]

            if id in id_set:
                return None

            tags = list(set(r["tags"]) & {"relevant", "irrelevant"})

            if not tags:
                return None

            id_set.add(id)

            r = r.copy()
            r["tags"] = tags
            return r

        rs = list(filter(None, map(map_f, rs)))

        tag_ctr = collections.Counter()

        for r in rs:
            tag_ctr.update(get_extended_tags(r))

        for r in rs:
            r["tags"] += [tag for tag in get_extended_tags(r) if tag_ctr[tag] > 1]

        print("Saving tweets to LabeledStatuses_Power_A_Excel.csv...")

        with ExcelCSVMarshaller("LabeledStatuses_Power_A_Excel.csv") as marshaller:
            for r in rs:
                marshaller.add(r)

        print("Saving tweets to LabeledStatuses_Power_A_Unix.csv...")

        with UnixCSVMarshaller("LabeledStatuses_Power_A_Unix.csv") as marshaller:
            for r in rs:
                marshaller.add(r)

        print("Saving tweets to LabeledStatuses_Power_A_AutoML.csv...")

        with open("LabeledStatuses_Power_A_AutoML.csv", "w", newline = "", encoding = "utf-8") as fd:
            csvw = csv.writer(fd)

            for r in rs:
                csvw.writerow([getnicetext(r), "relevant" if "relevant" in r["tags"] else "irrelevant"])

        print("Saving tweets to LabeledStatuses_Power_A.pkl...")

        with PickleMarshaller("LabeledStatuses_Power_A.pkl") as marshaller:
            for r in rs:
                marshaller.add(r)

        print("Saving tweets to LabeledStatuses_Power_A.bson...")

        with BSONMarshaller("LabeledStatuses_Power_A.bson") as marshaller:
            for r in rs:
                marshaller.add(r)

        print("Saving tweets to mongo://da1.eecs.utk.edu/twitter/LabeledStatuses_Power_A...")

        db.drop_collection("LabeledStatuses_Power_A")

        with opencoll(db, "LabeledStatuses_Power_A") as coll:
            coll.insert_many(rs, ordered = False)
