import csv
import collections
import datetime
import itertools
import math
import multiprocessing as mp
import os
import re
import sys
import traceback

#import tweepy
import nltk

from db import CityAreaDB, GeolocationDB
from match import StreetAddress, STATE_INITIALS
from sync import Channel

class CSV_SCRAPY(csv.Dialect):
    delimiter = "|"
    quotechar = "'"
    doublequote = True
    skipinitialspace = False
    lineterminator = "\n"
    quoting = csv.QUOTE_MINIMAL

GEODB  = GeolocationDB.open("geolocations.dat")
AREADB = CityAreaDB.open("cityareas.dat")

# This might be useful later if we use stderr for something
'''
def logfile_loop(logname, chan):
    logfd = sys.stderr if logname is None else open(logname, "a")

    def log_printf(string, *args):
        logfd.write("[%s] %s: %s\n" % (
                    datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
                    pname,
                    string % args
        ))

    while True:
        msg = chan.recv()

        if not chan or msg is None:
            break

        pname, data = msg

        if isinstance(msg, Exception):
            log_printf("Fatal %s exception raised, terminating", str(type(msg)))
        else:
            log_printf(str(msg))

    if logname is not None:
        logfd.close()
'''

def process(ifname, ofname):
    ifd = open(ifname, "r", newline = "")
    ofd = open(ofname, "w", newline = "")

    icsv = csv.DictReader(ifd, dialect = CSV_SCRAPY, quoting = csv.QUOTE_ALL)

    pname = ifd.name + " -> " + ofd.name

    ocsv = None

    for irow in icsv:
        if ocsv is None:
            ocsv = csv.DictWriter(ofd, dialect = CSV_SCRAPY, fieldnames = icsv.fieldnames + ["address", "latitude", "longitude", "latlongerr"])
            ocsv.writeheader()

        orow = dict(irow)

        orow["address"]    = None
        orow["latitude"]   = None
        orow["longitude"]  = None
        orow["latlongerr"] = None

        addr = StreetAddress.nlp(irow["text"])
        if addr is None:
            addr = StreetAddress.re(irow["text"])

        if addr is not None:
            orow["address"] = str(addr)

            query = {"country": "USA"}

            if addr.street is not None:
                query["street"] = addr.house_number + " " + addr.street

            if addr.city is not None:
                query["city"] = addr.city

            if addr.state is not None:
                query["state"] = STATE_INITIALS[addr.state]

            if addr.zip_code is not None:
                query["postalcode"] = addr.zip_code

            coord = GEODB.get(**query)
            if coord is not None:
                orow["latitude"], orow["longitude"] = coord

                if addr.street is not None:
                    # If a street address is specified, assume the
                    # coordinates are 100% precise
                    orow["latlongerr"] = 0.0
                else:
                    area = AREADB.get(addr.city, STATE_INITIALS[addr.state])

                    if area is not None:
                        # Calculate the radius of the city from its
                        # area. For simplicity, assume a perfectly
                        # circular city area
                        orow["latlongerr"] = math.sqrt(area / math.pi)

        ocsv.writerow(orow)

        if orow["latitude"] is not None:
            sys.stderr.write("Mapped tweet %s to coordinates (%f, %f)\n" % (
                orow["ID"],
                orow["latitude"],
                orow["longitude"]
            ))

        ifd.close()
        ofd.close()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage:", sys.argv[0], "[DATA dir]", file = sys.stderr)
        exit(-1)
    elif not os.path.isdir(sys.argv[1]):
        print("Usage:", sys.argv[0], "[DATA dir]", file = sys.stderr)
        exit(-1)

    with mp.Pool(processes = 4) as pool:
        for dirpath, _, filenames in os.walk(sys.argv[1]):
            for filename in filenames:
                ifname = os.path.join(dirpath, filename)
                rdot = ifname.rfind('.')

                if ifname[rdot:] != ".txt" or "(with location tags)" in ifname:
                    continue

                ofname = ifname[:rdot] + " (with location tags)" + ifname[rdot:]

                pool.apply_async(process, (ifname, ofname))

GEODB.close()
AREADB.close()
