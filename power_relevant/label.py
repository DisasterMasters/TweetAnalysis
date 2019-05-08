import sys
import re
import os

from common import getnicetext
from marshalling import *

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: " + sys.argv[0] + " <Pickled data>", file = sys.stderr)
        exit(-1)

    filename = sys.argv[1]
    ext = filename[filename.rfind("."):]

    rs = []

    cls = {
        ".csv": UnixCSVUnmarshaller,
        ".pkl": PickleUnmarshaller,
        ".bson": BSONUnmarshaller
    }[ext]

    with cls(filename) as unmarshaller:
        for r in unmarshaller:
            try:
                #if "tags" in r and ("relevant" in r["tags"] or "irrelevant" in r["tags"]):
                #    continue

                print("\n" + getnicetext(r) + "\n")
                relevant = input("\033[1mIs this relevant?\033[0m (yes/no/\033[4ms\033[0mkip) ").lower().strip()

                if relevant in {"yes", "y", "true", "t", "on"}:
                    r["tags"] = ["relevant"]
                    rs.append(r)
                elif relevant in {"no", "n", "false", "f", "off"}:
                    r["tags"] = ["irrelevant"]
                    rs.append(r)
            except KeyboardInterrupt:
                break

    if rs:
        os.rename(filename, filename + ".old")

        cls = {
            ".csv": UnixCSVMarshaller,
            ".pkl": PickleMarshaller,
            ".bson": BSONMarshaller
        }[ext]

        with cls(filename) as marshaller:
            for r in rs:
                marshaller.add(r)
