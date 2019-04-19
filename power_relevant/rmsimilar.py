import contextlib
import re
import sys

from fuzzywuzzy import process as fuzzy_process

if __name__ == "__main__":
    if len(sys.argv) > 1:
        threshold = int(sys.argv[1])
    else:
        threshold = 95

    regex = re.compile(r"\w+")
    dups = set()

    with opentunnel(), opendb() as db, opencoll(db, "LabeledStatuses_Power_A") as coll, PickleMarshaller("Removed.pkl") as marshaller:
        with contextlib.closing(coll.find(projection = ["id", "text"], no_cursor_timeout = True)) as cursor:
            try:
                for r in cursor:
                    if r["_id"] in dups:
                        continue

                    search = list(coll.find({"$text": {"$search": " ".join(regex.findall(r["text"]))}}))

                    for match, dist in fuzzy_process.extractOne(r, search, limit = 25, processor = lambda r: r["text"]):
                        if dist < threshold:
                            break

                        if match["id"] == r["id"]:
                            continue

                        dups.add(match["_id"])
                        marshaller.add(match)
            except KeyboardInterrupt:
                pass

        coll.delete_many({"_id": {"$in": list(dups)}})

    print("Removed %d duplicate records, see Removed.pkl for deleted records")
