import sys

from fuzzywuzzy import process as fuzzy_process

from common import getnicetext
from marshalling import *

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: " + sys.argv[0] + " <Pickled data>", file = sys.stderr)
        exit(-1)

    filename = sys.argv[1]
    ext = filename[filename.rfind("."):]

    cls = {
        ".csv": UnixCSVUnmarshaller,
        ".pkl": PickleUnmarshaller,
        ".bson": BSONUnmarshaller
    }[ext]

    with cls(filename) as unmarshaller:
        rs = list(unmarshaller)

    try:
        while True:
            query = {"text": input("Enter a string to search for: ").rstrip("\n")}

            matches = fuzzy_process.extract(query, rs, limit = 10, processor = getnicetext)

            for i, (r, dist) in enumerate(matches):
                print("%d (%d%% match):\n%s\n" % (i, dist, getnicetext(r)))

            try:
                query = int(input("Choose a match [0-9]: "))
            except ValueError:
                continue

            r = matches[query][0]
            print("Current tags: " + " ".join(r.get("tags", [])))

            relevant = input("Is this relevant? (yes/no/remove/\033[4ms\033[0mkip) ").lower().strip()

            if relevant in {"yes", "y", "true", "t", "on"}:
                r["tags"] = ["relevant"]
            elif relevant in {"no", "n", "false", "f", "off"}:
                r["tags"] = ["irrelevant"]
            elif relevant in {"remove", "delete", "erase", "rm"}:
                rs.remove(r)
    except KeyboardInterrupt:
        pass

    cls = {
        ".csv": UnixCSVMarshaller,
        ".pkl": PickleMarshaller,
        ".bson": BSONMarshaller
    }[ext]

    with cls(filename) as marshaller:
        for r in rs:
            marshaller.add(r)
