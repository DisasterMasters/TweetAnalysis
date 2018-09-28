# Uncomment for Python 2
#from __future__ import *

import time
import pickle
import urllib.request as urllib
import re

from lxml.html import parse
from lxml.etree import tostring
from geopy.geocoders import Nominatim
from geopy.exc import GeopyError

class OpenCtx:
    def __init__(self, db_cls, filename):
        self.db = db_cls()
        self.filename = filename

        try:
            fd = open(filename, "rb")

            self.db.load(fd)

            fd.close()
        except IOError:
            pass
        except EOFError:
            pass

    def __enter__(self):
        return self.db

    def __exit__(self, type, value, tb):
        fd = open(self.filename, "wb")

        self.db.dump(fd)

        fd.close()

class DBMixin:
    @classmethod
    def open(cls, filename):
        return OpenCtx(cls, filename)

    def __init__(self):
        self.cache = {}

    def load(self, fd):
        self.cache = pickle.load(fd)

    def dump(self, fd):
        pickle.dump(self.cache, fd)

    def loads(self, bs):
        self.cache = pickle.loads(bs)

    def dumps(self):
        return pickle.dumps(self.cache)

class CityAreaDB(DBMixin):
    TAGS_REGEX = re.compile(r'\<.+?\>')
    SQMI_REGEX = re.compile(r'Land area:\s*(?P<sqmi>[0-9\.]+)\s*square miles\.')

    def __init__(self):
        super().__init__()

    def get_area(self, city, state):
        key = (city, state)

        if key not in self.cache:
            try:
                htm = parse("http://www.city-data.com/city/%s-%s.html" % key)
            except OSError:
                return None

            time.sleep(1)

            for tag in htm.xpath('//section[@id="population-density"]'):
                text = CityAreaDB.TAGS_REGEX.sub('', tostring(tag).decode())
                match = CityAreaDB.SQMI_REGEX.search(text)

                if match is not None:
                    # Convert to km^2
                    self.cache[key] = float(match.group("sqmi")) * 2.589988

        return self.cache[key]

class GeolocationDB(DBMixin):
    def __init__(self):
        super().__init__()

        self.backend = Nominatim(
            user_agent = "curent-utk",
            country_bias = "USA"
        )

    def get_coords(self, **kwargs):
        key = tuple(sorted(kwargs.items()))

        if key not in self.cache:
            try:
                coord = self.backend.geocode(kwargs)

                # Nominatim allows at most one request per second
                # <https://operations.osmfoundation.org/policies/nominatim/>
                time.sleep(1)
            except GeopyError:
                coord = None

            if coord is None:
                return None

            self.cache[key] = (coord.latitude, coord.longitude)

        return self.cache[key]
