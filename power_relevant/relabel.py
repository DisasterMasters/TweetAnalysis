import re
import sys

from fuzzywuzzy import process as fuzzy_process

from common import *
from marshalling import *

if __name__ == "__main__":
    filename = sys.argv[1]
    ext = filename[filename.rfind("."):]
    regex = re.compile(r"\w+")

    try:
        cls = {
            ".csv": UnixCSVUnmarshaller,
            ".pkl": PickleUnmarshaller,
            ".bson": BSONUnmarshaller
        }[ext]

        with cls(filename) as marshaller:
            old_rs = list(marshaller)
    except FileNotFoundError:
        old_rs = []

    rs = []

    with opentunnel(), opendb() as db, opencoll(db, "LabeledStatuses_Power_A") as coll:
        while True:
            try:
                query = input("Enter a string to search for: ").rstrip("\n")

                matches = list(coll.find({"$text": {"$search": " ".join(regex.findall(query))}})) + old_rs
                matches = fuzzy_process.extract({"text": query}, matches, limit = 10, processor = getnicetext)

                for i, (r, dist) in enumerate(matches):
                    print("%d (%d%% match):\n%s\n" % (i, dist, getnicetext(r)))

                try:
                    query = int(input("Choose a match [0-9]: "))
                except ValueError:
                    continue

                r = matches[query][0]
                print("Current tags: " + " ".join(r.get("tags", [])))

                relevant = input("Is this relevant? (yes/no/\033[4ms\033[0mkip) ").lower().strip()

                if relevant in {"yes", "y", "true", "t", "on"}:
                    r["tags"] = ["relevant"]
                    rs.append(r)
                elif relevant in {"no", "n", "false", "f", "off"}:
                    r["tags"] = ["irrelevant"]
                    rs.append(r)
            except KeyboardInterrupt:
                break

    if rs:
        if old_rs:
            rs += old_rs

        cls = {
            ".csv": UnixCSVMarshaller,
            ".pkl": PickleMarshaller,
            ".bson": BSONMarshaller
        }[ext]

        with cls(filename) as marshaller:
            for r in rs:
                marshaller.add(r)
