import math

def geojson_to_coords(geojson):
    if geojson["type"] == "Point":
        [lon, lat] = r["coordinates"]
        err = 0.0

    elif geojson["type"] == "Polygon":
        cos_phi = math.sqrt(math.pi / 2)
        sec_phi = math.sqrt(2 / math.pi)
        R_earth = 6371.0

        # Map polygon coordinates to points on an equal-area cylindrical projection (Smyth equal-surface)
        polygons = [[(math.radians(lon) * cos_phi * R_earth, math.sin(math.radians(lat)) * sec_phi * R_earth) for [lat, lon] in p] for p in geojson["coordinates"]]

        areas = []
        comxs = []
        comys = []

        for p in polygons:
            segments = list(zip(p, p[1:] + p[:1]))

            a = math.fsum(x0 * y1 - x1 * y0 for (x0, y0), (x1, y1) in segments) * 0.5

            if a == 0.0:
                # Area described is a point, so return it
                x, y = p[0]

            else:
                x = math.fsum((x0 + x1) * (x0 * y1 - x1 * y0) for (x0, y0), (x1, y1) in segments) / (6 * a)
                y = math.fsum((y0 + y1) * (x0 * y1 - x1 * y0) for (x0, y0), (x1, y1) in segments) / (6 * a)

            areas.append(abs(a))
            comxs.append(x)
            comys.append(y)

        total_area = sum(areas)

        if total_area == 0.0:
            cx = math.fsum(comxs) / len(comxs)
            cy = math.fsum(comys) / len(comys)
        else:
            # Compute centroid of all polygons from weighted polygon centroids
            cx = math.fsum(c * a for c, a in zip(comxs, areas)) / total_area
            cy = math.fsum(c * a for c, a in zip(comys, areas)) / total_area

        # Unmap from projection to exact coordinates
        lat = math.degrees(math.asin(cy * (cos_phi / R_earth)))
        lon = math.degrees(cx * (sec_phi / R_earth))

        err = math.sqrt(total_area / math.pi)

    else:
        lat = None
        lon = None
        err = None

    return (lat, lon, err)
