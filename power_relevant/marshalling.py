import base64
import contextlib
import csv
import pickle

import bson

from common import getnicetext

def bson85_pack(r):
    return base64.b85encode(bson.BSON.encode(r)).decode("ascii")

def bson85_unpack(s):
    return bson.BSON.decode(base64.b85decode(s.encode("ascii")))

class ExcelCSVMarshaller:
    def __init__(self, filename):
        self.name = filename

    def __enter__(self):
        self.fd = open(self.name, "w", newline = "", encoding = "latin-1", errors = "replace")
        self.csvw = csv.writer(self.fd, dialect = csv.excel)

        self.csvw.writerow(["ID", "Text", "Relevant?", "Username", "Date", "BSON85"])

        return self

    def __exit__(self, type, value, traceback):
        self.fd.close()

    def add(self, r):
        if "tags" in r:
            if "relevant" in r["tags"]:
                relevant = "Yes"
            elif "irrelevant" in r["tags"]:
                relevant = "No"
            else:
                relevant = "Unknown"
        else:
            relevant = "Unknown"

        self.csvw.writerow([
            r["id_str"],
            getnicetext(r).replace("\n", " "),
            relevant,
            "@" + r["user"]["screen_name"],
            r["created_at"].isoformat(timespec = "seconds"),
            bson85_pack(r)
        ])

class UnixCSVMarshaller:
    def __init__(self, filename):
        self.name = filename

    def __enter__(self):
        self.fd = open(self.name, "w", newline = "", encoding = "utf-8")
        self.csvw = csv.writer(self.fd, dialect = csv.unix_dialect)

        self.csvw.writerow(["ID", "Text", "Relevant?", "Username", "Date", "BSON85"])

        return self

    def __exit__(self, type, value, traceback):
        self.fd.close()

    def add(self, r):
        if "tags" in r:
            if "relevant" in r["tags"]:
                relevant = "Yes"
            elif "irrelevant" in r["tags"]:
                relevant = "No"
            else:
                relevant = "Unknown"
        else:
            relevant = "Unknown"

        self.csvw.writerow([
            r["id_str"],
            getnicetext(r).replace("\n", " "),
            relevant,
            "@" + r["user"]["screen_name"],
            r["created_at"].isoformat(timespec = "seconds"),
            bson85_pack(r)
        ])

class PickleMarshaller:
    def __init__(self, filename):
        self.name = filename

    def __enter__(self):
        self.fd = open(self.name, "wb")
        self.pickler = pickle.Pickler(self.fd)

        return self

    def __exit__(self, type, value, traceback):
        self.fd.close()

    def add(self, r):
        self.pickler.dump(r)

class BSONMarshaller:
    def __init__(self, filename):
        self.name = filename

    def __enter__(self):
        self.fd = open(self.name, "wb")

        return self

    def __exit__(self, type, value, traceback):
        self.fd.close()

    def add(self, r):
        self.fd.write(bson.BSON.encode(r))

class ExcelCSVUnmarshaller:
    def __init__(self, filename):
        self.name = filename

    def __enter__(self):
        self.fd = open(self.name, "r", newline = "", encoding = "latin-1")
        return self

    def __exit__(self, type, value, traceback):
        self.fd.close()

    def __iter__(self):
        for row in csv.DictReader(self.fd, dialect = csv.excel):
            r = bson85_unpack(row["BSON85"])

            if "tags" in r:
                try:
                    r["tags"].remove("relevant")
                except ValueError:
                    pass

                try:
                    r["tags"].remove("irrelevant")
                except ValueError:
                    pass

                relevant = row["Relevant?"].lower().strip()

                if relevant in {"yes", "y", "true", "t", "on"}:
                    r["tags"].append("relevant")
                elif relevant in {"no", "n", "false", "f", "off"}:
                    r["tags"].append("irrelevant")

            yield r

class UnixCSVUnmarshaller:
    def __init__(self, filename):
        self.name = filename

    def __enter__(self):
        self.fd = open(self.name, "r", newline = "", encoding = "utf-8")
        return self

    def __exit__(self, type, value, traceback):
        self.fd.close()

    def __iter__(self):
        for row in csv.DictReader(self.fd, dialect = csv.unix_dialect):
            r = bson85_unpack(row["BSON85"])
            relevant = row["Relevant?"].lower().strip()

            if "tags" in r:
                try:
                    r["tags"].remove("relevant")
                except ValueError:
                    pass

                try:
                    r["tags"].remove("irrelevant")
                except ValueError:
                    pass
            else:
                r["tags"] = []

            if relevant in {"yes", "y", "true", "t", "on"}:
                r["tags"].append("relevant")
            elif relevant in {"no", "n", "false", "f", "off"}:
                r["tags"].append("irrelevant")

            yield r

class PickleUnmarshaller:
    def __init__(self, filename):
        self.name = filename

    def __enter__(self):
        self.fd = open(self.name, "rb")
        self.pickler = pickle.Unpickler(self.fd)

        return self

    def __exit__(self, type, value, traceback):
        self.fd.close()

    def __iter__(self):
        while True:
            try:
                yield self.pickler.load()
            except EOFError:
                break

class BSONUnmarshaller:
    def __init__(self, filename):
        self.name = filename

    def __enter__(self):
        self.fd = open(self.name, "rb")

        return self

    def __exit__(self, type, value, traceback):
        self.fd.close()

    def __iter__(self):
        yield from bson.decode_file_iter(self.fd)
