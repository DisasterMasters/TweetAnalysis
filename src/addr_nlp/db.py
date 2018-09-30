import abc
import time
import pickle
import urllib.request as urllib
import re

from lxml.html import parse
from lxml.etree import tostring
from geopy.geocoders import Nominatim
from geopy.exc import GeopyError

from sync import RWLock

class DB(abc.ABC):
    @classmethod
    def open(cls, filename):
        db = cls()
        db.filename = filename

        try:
            with open(filename, "rb") as fd:
                db.load(fd)
        except Exception:
            db.cache.clear()

        return db

    def __init__(self):
        self.cache = {}
        self.filename = None
        self.rwlock = RWLock()

    def __enter__(self):
        return self

    def __exit__(self, type, value, tb):
        self.close()

    def load(self, fd):
        try:
            self.rwlock.acquire_write()

            self.cache = pickle.load(fd)
        finally:
            self.rwlock.release()

    def dump(self, fd):
        try:
            self.rwlock.acquire_read()

            pickle.dump(self.cache, fd)
        finally:
            self.rwlock.release()

    def loads(self, bs):
        try:
            self.rwlock.acquire_write()

            self.cache = pickle.loads(bs)
        finally:
            self.rwlock.release()

    def dumps(self):
        try:
            self.rwlock.acquire_read()

            return pickle.dumps(self.cache)
        finally:
            self.rwlock.release()

    def close(self):
        try:
            self.rwlock.acquire_read()

            if self.filename is not None:
                with open(self.filename, "wb") as fd:
                    self.dump(fd)
        finally:
            self.rwlock.release()

    @abc.abstractmethod
    def update(self, *args, **kwargs):
        pass

    def get(self, *args, **kwargs):
        k = args + tuple(sorted(kwargs.items()))

        try:
            self.rwlock.acquire_read()

            if k not in self.cache:
                self.rwlock.promote()

                v = self.update(*args, **kwargs)

                if v is None:
                    return None
                else:
                    self.cache[k] = v

            return self.cache[k]
        finally:
            self.rwlock.release()

class CityAreaDB(DB):
    TAGS_REGEX = re.compile(r'\<.+?\>')
    SQMI_REGEX = re.compile(r'Land area:\s*(?P<sqmi>[0-9\.]+)\s*square miles\.')

    def update(self, *args, **kwargs):
        city, state = args

        try:
            htm = parse("http://www.city-data.com/city/%s-%s.html" % args)
        except OSError:
            return None

        time.sleep(1)

        for tag in htm.xpath('//section[@id="population-density"]'):
            text = CityAreaDB.TAGS_REGEX.sub('', tostring(tag).decode())
            match = CityAreaDB.SQMI_REGEX.search(text)

            if match is not None:
                # Convert to km^2
                return float(match.group("sqmi")) * 2.589988

        return None

class GeolocationDB(DB):
    def __init__(self):
        super().__init__()

        self.backend = Nominatim(
            user_agent = "curent-utk",
            country_bias = "USA"
        )

    def update(self, *args, **kwargs):
        try:
            coord = self.backend.geocode(kwargs)
        except GeopyError:
            return None

        # Nominatim allows at most one request per second
        # <https://operations.osmfoundation.org/policies/nominatim/>
        time.sleep(1)

        return None if coord is None else (coord.latitude, coord.longitude)
