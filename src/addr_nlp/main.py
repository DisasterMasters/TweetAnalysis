import csv
import collections
import itertools
import math
import os
import re
import sys
import time

#import tweepy
import nltk

from db import CityAreaDB, GeolocationDB
from match import StreetAddress, STATE_INITIALS

class CSV_SCRAPY(csv.Dialect):
    delimiter = "|"
    quotechar = "'"
    doublequote = True
    skipinitialspace = False
    lineterminator = "\n"
    quoting = csv.QUOTE_MINIMAL

GEODB  = GeolocationDB.open("geolocations.dat")
AREADB = CityAreaDB.open("cityareas.dat")

def process_csv(ifd, ofd, log = sys.stderr):
    icsv = csv.DictReader(ifd, dialect = CSV_SCRAPY, quoting = csv.QUOTE_ALL)

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

        if log is not None and orow["address"] is not None:
            log.write("%s -> %s: \"%s\" matched in tweet \"%s\"\n" % (ifd.name, ofd.name, orow["text"], orow["address"]))

            if orow["latitude"] is not None and orow["longitude"] is not None:
                log.write("%s -> %s: \"%s\" mapped to coordinates (%f, %f)" % (ifd.name, ofd.name, orow["address"], orow["latitude"], orow["longitude"]))

                if orow["latlongerr"] is not None:
                    log.write(" within a %f km radius\n" % orow["latlongerr"])
                else:
                    log.write(" within an unknown radius\n")

        ocsv.writerow(orow)

LOGNAME = "postal_regex.log"

if __name__ == "__main__":
    if len(sys.argv) > 1 and os.path.isdir(sys.argv[1]):
        for dirpath, dirnames, filenames in os.walk(sys.argv[1]):
            for ifname in filenames:
                rdot = ifname.rfind('.')

                if ifname[rdot:] != ".txt":
                    continue

                ofname = ifname[:rdot] + "_W_LOCATION_TAGS" + ifname[rdot:]

                with open(LOGNAME, "a") as log:
                    try:
                        with open(ifname, "r", newline = '') as ifd:
                            with open(ofname, "w", newline = '') as ofd:
                                process_csv(ifd, ofd)
                    except Exception as err:
                        log.write("%s -> %s: Uncaught exception of type %s\n" % (ifd.name, ofd.name, str(type(err))))

                        for line in traceback.format_exc().splitlines():
                            log.write("%s -> %s: %s\n" % (ifd.name, ofd.name, line))

                        log.flush()
    else:
        try:
            ifd = open(sys.argv[1], "r", newline = '') if len(sys.argv) > 1 else sys.stdin
            ofd = open(sys.argv[2], "w", newline = '') if len(sys.argv) > 2 else sys.stdout

            process_csv(ifd, ofd)
        finally:
            if len(sys.argv) > 1:
                ifd.close()
            if len(sys.argv) > 2:
                ofd.close()

GEODB.close()
AREADB.close()
