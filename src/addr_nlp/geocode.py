import time
import shelve

import geopy.geocoders
import geopy.exc

class GeolocationDB:
    def __init__(self, filename):
        self.db = shelve.open(filename, "c")

        self.api = geopy.geocoders.Nominatim(
            user_agent = "curent-utk",
            country_bias = "USA"
        )

        self.dt = float('-inf')

    def __enter__(self):
        return self

    def __exit__(self, type, value, traceback):
        self.db.close()

    def __getitem__(self, key):
        if key.raw not in self.db:
            query = {"country": "USA"}

            if key.house_number is not None and key.street is not None:
                query["street"] = key.house_number + " " + key.street

            if key.city is not None:
                query["city"] = key.city

            if key.state is not None:
                query["state"] = key.state

            if key.zip_code is not None:
                query["postalcode"] = key.zip_code

            # Nominatim allows at most one request per second
            # <https://operations.osmfoundation.org/policies/nominatim/>
            dt = time.perf_counter()

            if dt - self.dt < 1.1:
                time.sleep(1.1 - dt + self.dt)

            self.dt = dt

            try:
                coord = self.api.geocode(query, geometry = "geojson")
            except geopy.exc.GeopyError:
                return None

            self.db[key.raw] = coord.raw if coord is not None else None

        return self.db[key.raw]
