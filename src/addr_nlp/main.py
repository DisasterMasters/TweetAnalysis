import collections
import contextlib
import datetime
import math
import threading
import os
import sys
import queue
import enum

import nltk

from db import GeolocationDB, dumps
from geojson import geojson_to_coords
from match import StreetAddress

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage:", sys.argv[0], "<input collection> <output collection>", file = sys.stderr)
        exit(-1)

    ctr = collections.Counter()

    with contextlib.closing(GeolocationDB("geolocations")) as geodb, contextlib.closing(pymongo.MongoClient()) as conn:
        def get_coord_info(r):
            if r["coordinates"] is not None:
                geojson = r["coordinates"]
            elif r["place"] is not None:
                geojson = r["place"]["bounding_box"]
            else:
                try:
                    text = r["extended_tweet"]["full_text"]
                except KeyError:
                    text = r["text"]

                addr = StreetAddress.nlp(text)
                if addr is not None:
                    #msgq.put(Signal.tweet_addr_nlp)
                else:
                    addr = StreetAddress.re(text)
                    if addr is not None:
                        #msgq.put(Signal.tweet_addr_re)
                    else:
                        addr = StreetAddress.statemap(text)
                        if addr is not None:
                            #msgq.put(Signal.tweet_addr_statemap)
                        else:
                            return None

                db_loc = geodb[addr._replace(raw = None)]

                if "geojson" in db_loc.raw:
                    geojson = db_loc.raw["geojson"]
                else:
                    geojson = {
                        "type": "Point",
                        "coordinates": [db_loc.longitude, db_loc.latitude]
                    }

            lat, lon, err = geojson_to_coords(geojson)

            return {
                "id": r["id"],
                "latitude": lat,
                "longitude": lon,
                "error": err,
                "geojson": geojson
            }

        coll_in = conn["twitter"][sys.argv[1]]
        coll_out = conn["twitter"][sys.argv[2]]

        coll_out.create_index([('id', pymongo.HASHED)], name = 'id_index')
        coll_out.create_index([('id', pymongo.ASCENDING)], name = 'id_ordered_index')
        coll_out.create_index([('geojson', pymongo.GEOSPHERE)], name = 'geojson_index')

        results = list(filter(map(get_coord_info, coll_in.find())))

        coll_out.insert_many(results, ordered = False)

    msg = """
Results for %s -> %s:
--------------------------------------------------------------------------------
Total tweets: %d
Tweets that have geolocation info: %d
Tweets that have an address in their text: %d
Tweets whose address was extracted via nlp(): %d
Tweets whose address was extracted via re(): %d
Tweets whose address was extracted via statemap(): %d
Tweets whose address mapped to a valid geolocation: %d
--------------------------------------------------------------------------------
Tweets whose geolocation error is equal to 0 km: %d
Tweets whose geolocation error is in the range (0 km, 1 km]: %d
Tweets whose geolocation error is in the range (1 km, 5 km]: %d
Tweets whose geolocation error is in the range (5 km, 25 km]: %d
Tweets whose geolocation error is in the range (25 km, 100 km]: %d
Tweets whose geolocation error is greater than 100 km: %d
--------------------------------------------------------------------------------
Pickled collections.Counter of above statistics:
%s""" % (
        sys.argv[1], sys.argv[2],
        ctr["tweet"],
        ctr["tweet_geo"],
        ctr["tweet_addr"],
        ctr["tweet_addr_nlp"],
        ctr["tweet_addr_re"],
        ctr["tweet_addr_statemap"],
        ctr["tweet_geo_fromaddr"],
        ctr["tweet_geo_0km"],
        ctr["tweet_geo_0to1km"],
        ctr["tweet_geo_1to5km"],
        ctr["tweet_geo_5to25km"],
        ctr["tweet_geo_25to100km"],
        ctr["tweet_geo_gt100km"],
        dumps(ctr).decode()
    )

    print(msg)
