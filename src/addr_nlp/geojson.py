import math
import sys

# Constants
cos_phi = math.sqrt(math.pi / 2)
sec_phi = math.sqrt(2 / math.pi)
R_earth = 6371.0

# Map latitude and longitude to points on a Smyth cylindrical projection.
#
# This needs to be done because an equirectangular projection (i.e. using
# longitude and latitude as horizontal and vertical points directly) is not
# equal area, and points closer to the poles are disproportionately large
# compared to points near the equator.
def smyth_map(lat, lon):
    return (math.radians(lon) * cos_phi * R_earth, math.sin(math.radians(lat)) * sec_phi * R_earth)

# Reverse mapping of smyth_map()
def smyth_unmap(x, y):
    return (math.degrees(math.asin(y * (cos_phi / R_earth))), math.degrees(x * (sec_phi / R_earth)))

def geojson_point(coordinates):
    return (coordinates[1], coordinates[0], 0.0)

def geojson_multipoint(coordinates):
    points = [smyth_map(lat, lon) for [lon, lat] in coordinates]

    # Should be fine to use latitude/longitude directly here
    lat = math.fsum(lat for [lon, lat] in coordinates) / len(coordinates)
    lon = math.fsum(lon for [lon, lat] in coordinates) / len(coordinates)

    cx = math.fsum(x for x, y in points) / len(points)
    cy = math.fsum(y for x, y in points) / len(points)
    err = max(math.hypot(x - cx, y - cy) for x, y in points)

    return (lat, lon, err)

def geojson_multipolygon(coordinates):
    # Calculate the centroid and area of an arbitrary polygon
    def get_centroid_area(p):
        segments = list(zip(p, p[1:] + p[:1]))

        a = math.fsum(x0 * y1 - x1 * y0 for (x0, y0), (x1, y1) in segments) * 0.5

        if a == 0.0:
            # Area described is a point, so return it
            x, y = p[0]

        else:
            x = math.fsum((x0 + x1) * (x0 * y1 - x1 * y0) for (x0, y0), (x1, y1) in segments) / (6 * a)
            y = math.fsum((y0 + y1) * (x0 * y1 - x1 * y0) for (x0, y0), (x1, y1) in segments) / (6 * a)

        return (x, y, abs(a))

    points = []
    coms = []

    for polygon in coordinates:
        current_points = [[smyth_map(lat, lon) for [lon, lat] in p] for p in polygon]
        current_coms = list(map(get_centroid_area, current_points))

        # In GeoJSON, all polygons after the first one are holes. Give these holes a negative area.
        points += current_points[0]
        coms += current_coms[:1] + [(x, y, -a) for x, y, a in current_coms[1:]]

    total_area = math.fsum(a for x, y, a in coms)

    # Compute centroid of all polygons from weighted polygon centroids
    if total_area == 0.0:
        cx = math.fsum(x for x, y, a in coms) / len(coms)
        cy = math.fsum(y for x, y, a in coms) / len(coms)
    else:
        cx = math.fsum(x * a for x, y, a in coms) / total_area
        cy = math.fsum(y * a for x, y, a in coms) / total_area

    lat, lon = smyth_unmap(cx, cy)
    err = max(math.hypot(x - cx, y - cy) for x, y in points)

    return (lat, lon, err)

def geojson_polygon(coordinates):
    return geojson_multipolygon([coordinates])

def geojson_to_coords(geojson):
    geojson_f = {
        "Point": geojson_point,
        "MultiPoint": geojson_multipoint,
        "Polygon": geojson_polygon,
        "MultiPolygon": geojson_multipolygon
    }.get(geojson["type"], lambda _: (None, None, None))

    return geojson_f(geojson["coordinates"])
