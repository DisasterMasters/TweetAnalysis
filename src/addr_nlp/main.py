import csv
import collections
import datetime
import math
import threading
import os
import sys
import queue
import enum

import nltk

from db import CityAreaDB, GeolocationDB, loads, dumps
from match import StreetAddress

LOG_NAME = "addr_nlp.log"

Signal = enum.Enum("Signal", """
tweet
tweet_addr
tweet_addr_street
tweet_addr_city
tweet_addr_state
tweet_addr_re
tweet_addr_nlp
tweet_addr_statemap
tweet_coord
tweet_coord_lt1km
tweet_coord_lt5km
tweet_coord_lt25km
tweet_coord_lt100km
""")

class ScrapyDialect(csv.Dialect):
    delimiter = "|"
    quotechar = "'"
    doublequote = True
    skipinitialspace = False
    lineterminator = "\n"
    quoting = csv.QUOTE_MINIMAL

GEODB  = GeolocationDB("geolocations")
AREADB = CityAreaDB("cityareas")

def log_thread(dirpool, filepool, msgq, tee = True):
    counter = collections.defaultdict(int)

    for thread in dirpool:
        thread.join()

        try:
            with open(os.path.join(thread.name, LOG_NAME), "r") as oldlog:
                lastline = oldlog.readlines()[-1].strip()
        except FileNotFoundError:
            continue

        for k, v in loads(lastline.encode()).items():
            counter[k] += v

    logfd = open(os.path.join(threading.current_thread().name, LOG_NAME), "w")

    while any(thread.is_alive() for thread in filepool):
        try:
            msg = msgq.get(timeout = 2)
        except queue.Empty:
            continue

        if type(msg) is str:
            logfd.write(msg)
            if tee:
                sys.stdout.write(msg)
        else:
            counter[msg] += 1


    msg = """
Results for %s:
--------------------------------------------------------------------------------
Total tweets: %d
Tweets containing any address information in their text: %d
Tweets containing street-level address information in their text: %d
Tweets containing city-level address information in their text: %d
Tweets containing state-level address information in their text: %d
Tweets whose address information was extracted via nlp(): %d
Tweets whose address information was extracted via re(): %d
Tweets whose address information was extracted via statemap(): %d
--------------------------------------------------------------------------------
Tweets whose address mapped to a valid geolocation: %d
Tweets whose geolocation error is < 1 km: %d
Tweets whose geolocation error is < 5 km: %d
Tweets whose geolocation error is < 25 km: %d
Tweets whose geolocation error is < 100 km: %d
--------------------------------------------------------------------------------

%s""" % (
        dirpath,
        counter[Signal.tweet],
        counter[Signal.tweet_addr],
        counter[Signal.tweet_addr_street],
        counter[Signal.tweet_addr_city],
        counter[Signal.tweet_addr_state],
        counter[Signal.tweet_addr_nlp],
        counter[Signal.tweet_addr_re],
        counter[Signal.tweet_addr_statemap],
        counter[Signal.tweet_coord],
        counter[Signal.tweet_coord_lt1km],
        counter[Signal.tweet_coord_lt5km],
        counter[Signal.tweet_coord_lt25km],
        counter[Signal.tweet_coord_lt100km],
        dumps(counter).decode()
    )

    logfd.write(msg)
    if tee:
        sys.stdout.write(msg)

    logfd.close()

    GEODB.sync()
    AREADB.sync()

def bsv_thread(ifname, ofname, msgq):
    ifd = open(ifname, "r", newline = "")
    ofd = open(ofname, "w", newline = "")

    icsv = csv.DictReader(ifd, dialect = ScrapyDialect, quoting = csv.QUOTE_ALL)
    ocsv = None

    def send(msg):
        msgq.put("[%s] %s:\n%s\n" % (
            datetime.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
            threading.current_thread().name, msg
        ))

    for irow in icsv:
        msgq.put(Signal.tweet)

        # Can't get iscv.fieldnames until at least some input has been read
        if ocsv is None:
            ocsv = csv.DictWriter(
                ofd,
                dialect = ScrapyDialect,
                fieldnames = icsv.fieldnames +
                             ["address", "coord_lat", "coord_lon", "coord_err"]
                )
            ocsv.writeheader()

        orow = dict(irow)

        orow["address"]   = None
        orow["coord_lat"] = None
        orow["coord_lon"] = None
        orow["coord_err"] = None

        text = irow["Translation"] if "Translation" in irow else irow["text"]

        if type(text) is not str:
            continue

        addr = StreetAddress.nlp(text)
        if addr is not None:
            msgq.put(Signal.tweet_addr_nlp)
        else:
            addr = StreetAddress.re(text)
            if addr is not None:
                msgq.put(Signal.tweet_addr_re)
            else:
                addr = StreetAddress.statemap(text)
                if addr is not None:
                    msgq.put(Signal.tweet_addr_statemap)

        if addr is not None:
            send("Found \"%s\" in tweet \"%s\"" % (str(addr), text))
            msgq.put(Signal.tweet_addr)

            if addr.house_number is not None and addr.street is not None:
                msgq.put(Signal.tweet_addr_street)
            elif addr.city is not None:
                msgq.put(Signal.tweet_addr_city)
            elif addr.state is not None:
                msgq.put(Signal.tweet_addr_state)

            orow["address"] = str(addr)
            coord = GEODB[addr]

            if coord is not None:
                msgq.put(Signal.tweet_coord)
                orow["coord_lat"], orow["coord_lon"] = coord

                if addr.street is not None:
                    # If a street address is specified, assume the
                    # coordinates are 100% precise
                    orow["coord_err"] = 0.0
                else:
                    area = AREADB[(addr.city, addr.state)]

                    if area is not None:
                        # Calculate the radius of the city from its
                        # area. For simplicity, assume a perfectly
                        # circular city area
                        orow["coord_err"] = math.sqrt(area / math.pi)

                if orow["coord_err"] is not None:
                    if orow["coord_err"] <= 1.0:
                        msgq.put(Signal.tweet_coord_lt1km)
                    elif orow["coord_err"] <= 5.0:
                        msgq.put(Signal.tweet_coord_lt5km)
                    elif orow["coord_err"] <= 25.0:
                        msgq.put(Signal.tweet_coord_lt25km)
                    elif orow["coord_err"] <= 100.0:
                        msgq.put(Signal.tweet_coord_lt100km)

                send("Mapped \"%s\" to coordinates (%f, %f) within %s" % (
                    str(addr),
                    orow["coord_lat"],
                    orow["coord_lon"],
                    "an unknown radius" if orow["coord_err"] is None else "a " + str(orow["coord_err"]) + " km radius"
                ))

        # Prevent DictWriter from complaining about extra fields
        orow = {k: v for k, v in orow.items() if k in ocsv.fieldnames}

        ocsv.writerow(orow)

    ifd.close()
    ofd.close()

if __name__ == "__main__":
    if len(sys.argv) < 2 or not os.path.isdir(sys.argv[1]):
        print("Usage:", sys.argv[0], "[DATA dir]", file = sys.stderr)
        exit(-1)

    logthreads = {}

    for dirpath, dirnames, filenames in os.walk(sys.argv[1], topdown = False):
        filepool = []
        dirpool = [logthreads[os.path.join(dirpath, d)] for d in dirnames if os.path.join(dirpath, d) in logthreads]
        msgq = queue.Queue()

        for filename in filenames:
            ifname = os.path.join(dirpath, filename)
            rdot = ifname.rfind('.')

            if ifname[rdot:] != ".txt" or "_WCOORDS" in ifname:
                continue

            ofname = ifname[:rdot] + "_WCOORDS" + ifname[rdot:]

            # Note: doesn't actually do any multithreading because of GIL
            filepool.append(threading.Thread(
                name = ifname + " -> " + ofname,
                target = bsv_thread,
                args = (ifname, ofname, msgq)
            ))

            filepool[-1].start()

        logthreads[dirpath] = threading.Thread(
            name = dirpath,
            target = log_thread,
            args = (dirpool, filepool, msgq)
        )

        logthreads[dirpath].start()

    for v in logthreads.values():
        v.join()

GEODB.close()
AREADB.close()
