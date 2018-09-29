# Uncomment for Python 2
#from __future__ import *

import atexit
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

from city_area import CityAreaDB, GeolocationDB

# Obtained from <https://pe.usps.com/text/pub28/28apc_002.htm>
STREET_SUFFIXES = {
    'ALLEE', 'ALLEY', 'ALLY', 'ALY', 'ANEX', 'ANNEX', 'ANNX', 'ANX', 'ARC',
    'ARCADE', 'AV', 'AVE', 'AVEN', 'AVENU', 'AVENUE', 'AVN', 'AVNUE', 'BAYOO',
    'BAYOU', 'BCH', 'BEACH', 'BEND', 'BG', 'BGS', 'BLF', 'BLFS', 'BLUF',
    'BLUFF', 'BLUFFS', 'BLVD', 'BND', 'BOT', 'BOTTM', 'BOTTOM', 'BOUL',
    'BOULEVARD', 'BOULV', 'BR', 'BRANCH', 'BRDGE', 'BRG', 'BRIDGE', 'BRK',
    'BRKS', 'BRNCH', 'BROOK', 'BROOKS', 'BTM', 'BURG', 'BURGS', 'BYP', 'BYPA',
    'BYPAS', 'BYPASS', 'BYPS', 'BYU', 'CAMP', 'CANYN', 'CANYON', 'CAPE',
    'CAUSEWAY', 'CAUSWA', 'CEN', 'CENT', 'CENTER', 'CENTERS', 'CENTR',
    'CENTRE', 'CIR', 'CIRC', 'CIRCL', 'CIRCLE', 'CIRCLES', 'CIRS', 'CLB',
    'CLF', 'CLFS', 'CLIFF', 'CLIFFS', 'CLUB', 'CMN', 'CMNS', 'CMP', 'CNTER',
    'CNTR', 'CNYN', 'COMMON', 'COMMONS', 'COR', 'CORNER', 'CORNERS', 'CORS',
    'COURSE', 'COURT', 'COURTS', 'COVE', 'COVES', 'CP', 'CPE', 'CRCL', 'CRCLE',
    'CREEK', 'CRES', 'CRESCENT', 'CREST', 'CRK', 'CROSSING', 'CROSSROAD',
    'CROSSROADS', 'CRSE', 'CRSENT', 'CRSNT', 'CRSSNG', 'CRST', 'CSWY', 'CT',
    'CTR', 'CTRS', 'CTS', 'CURV', 'CURVE', 'CV', 'CVS', 'CYN', 'DALE', 'DAM',
    'DIV', 'DIVIDE', 'DL', 'DM', 'DR', 'DRIV', 'DRIVE', 'DRIVES', 'DRS', 'DRV',
    'DV', 'DVD', 'EST', 'ESTATE', 'ESTATES', 'ESTS', 'EXP', 'EXPR', 'EXPRESS',
    'EXPRESSWAY', 'EXPW', 'EXPY', 'EXT', 'EXTENSION', 'EXTENSIONS', 'EXTN',
    'EXTNSN', 'EXTS', 'FALL', 'FALLS', 'FERRY', 'FIELD', 'FIELDS', 'FLAT',
    'FLATS', 'FLD', 'FLDS', 'FLS', 'FLT', 'FLTS', 'FORD', 'FORDS', 'FOREST',
    'FORESTS', 'FORG', 'FORGE', 'FORGES', 'FORK', 'FORKS', 'FORT', 'FRD',
    'FRDS', 'FREEWAY', 'FREEWY', 'FRG', 'FRGS', 'FRK', 'FRKS', 'FRRY', 'FRST',
    'FRT', 'FRWAY', 'FRWY', 'FRY', 'FT', 'FWY', 'GARDEN', 'GARDENS', 'GARDN',
    'GATEWAY', 'GATEWY', 'GATWAY', 'GDN', 'GDNS', 'GLEN', 'GLENS', 'GLN',
    'GLNS', 'GRDEN', 'GRDN', 'GRDNS', 'GREEN', 'GREENS', 'GRN', 'GRNS', 'GROV',
    'GROVE', 'GROVES', 'GRV', 'GRVS', 'GTWAY', 'GTWY', 'HARB', 'HARBOR',
    'HARBORS', 'HARBR', 'HAVEN', 'HBR', 'HBRS', 'HEIGHTS', 'HIGHWAY', 'HIGHWY',
    'HILL', 'HILLS', 'HIWAY', 'HIWY', 'HL', 'HLLW', 'HLS', 'HOLLOW', 'HOLLOWS',
    'HOLW', 'HOLWS', 'HRBOR', 'HT', 'HTS', 'HVN', 'HWAY', 'HWY', 'INLET',
    'INLT', 'IS', 'ISLAND', 'ISLANDS', 'ISLE', 'ISLES', 'ISLND', 'ISLNDS',
    'ISS', 'JCT', 'JCTION', 'JCTN', 'JCTNS', 'JCTS', 'JUNCTION', 'JUNCTIONS',
    'JUNCTN', 'JUNCTON', 'KEY', 'KEYS', 'KNL', 'KNLS', 'KNOL', 'KNOLL',
    'KNOLLS', 'KY', 'KYS', 'LAKE', 'LAKES', 'LAND', 'LANDING', 'LANE', 'LCK',
    'LCKS', 'LDG', 'LDGE', 'LF', 'LGT', 'LGTS', 'LIGHT', 'LIGHTS', 'LK', 'LKS',
    'LN', 'LNDG', 'LNDNG', 'LOAF', 'LOCK', 'LOCKS', 'LODG', 'LODGE', 'LOOP',
    'LOOPS', 'MALL', 'MANOR', 'MANORS', 'MDW', 'MDWS', 'MEADOW', 'MEADOWS',
    'MEDOWS', 'MEWS', 'MILL', 'MILLS', 'MISSION', 'MISSN', 'ML', 'MLS', 'MNR',
    'MNRS', 'MNT', 'MNTAIN', 'MNTN', 'MNTNS', 'MOTORWAY', 'MOUNT', 'MOUNTAIN',
    'MOUNTAINS', 'MOUNTIN', 'MSN', 'MSSN', 'MT', 'MTIN', 'MTN', 'MTNS', 'MTWY',
    'NCK', 'NECK', 'OPAS', 'ORCH', 'ORCHARD', 'ORCHRD', 'OVAL', 'OVERPASS',
    'OVL', 'PARK', 'PARKS', 'PARKWAY', 'PARKWAYS', 'PARKWY', 'PASS', 'PASSAGE',
    'PATH', 'PATHS', 'PIKE', 'PIKES', 'PINE', 'PINES', 'PKWAY', 'PKWY',
    'PKWYS', 'PKY', 'PL', 'PLACE', 'PLAIN', 'PLAINS', 'PLAZA', 'PLN', 'PLNS',
    'PLZ', 'PLZA', 'PNE', 'PNES', 'POINT', 'POINTS', 'PORT', 'PORTS', 'PR',
    'PRAIRIE', 'PRK', 'PRR', 'PRT', 'PRTS', 'PSGE', 'PT', 'PTS', 'RAD',
    'RADIAL', 'RADIEL', 'RADL', 'RAMP', 'RANCH', 'RANCHES', 'RAPID',
    'RAPIDS', 'RD', 'RDG', 'RDGE', 'RDGS', 'RDS', 'REST', 'RIDGE', 'RIDGES',
    'RIV', 'RIVER', 'RIVR', 'RNCH', 'RNCHS', 'ROAD', 'ROADS', 'ROUTE', 'ROW',
    'RPD', 'RPDS', 'RST', 'RTE', 'RUE', 'RUN', 'RVR', 'SHL', 'SHLS', 'SHOAL',
    'SHOALS', 'SHOAR', 'SHOARS', 'SHORE', 'SHORES', 'SHR', 'SHRS', 'SKWY',
    'SKYWAY', 'SMT', 'SPG', 'SPGS', 'SPNG', 'SPNGS', 'SPRING', 'SPRINGS',
    'SPRNG', 'SPRNGS', 'SPUR', 'SPURS', 'SQ', 'SQR', 'SQRE', 'SQRS', 'SQS',
    'SQU', 'SQUARE', 'SQUARES', 'ST', 'STA', 'STATION', 'STATN', 'STN', 'STR',
    'STRA', 'STRAV', 'STRAVEN', 'STRAVENUE', 'STRAVN', 'STREAM', 'STREET',
    'STREETS', 'STREME', 'STRM', 'STRT', 'STRVN', 'STRVNUE', 'STS', 'SUMIT',
    'SUMITT', 'SUMMIT', 'TER', 'TERR', 'TERRACE', 'THROUGHWAY', 'TPKE',
    'TRACE', 'TRACES', 'TRACK', 'TRACKS', 'TRAFFICWAY', 'TRAIL', 'TRAILER',
    'TRAILS', 'TRAK', 'TRCE', 'TRFY', 'TRK', 'TRKS', 'TRL', 'TRLR', 'TRLRS',
    'TRLS', 'TRNPK', 'TRWY', 'TUNEL', 'TUNL', 'TUNLS', 'TUNNEL', 'TUNNELS',
    'TUNNL', 'TURNPIKE', 'TURNPK', 'UN', 'UNDERPASS', 'UNION', 'UNIONS', 'UNS',
    'UPAS', 'VALLEY', 'VALLEYS', 'VALLY', 'VDCT', 'VIA', 'VIADCT', 'VIADUCT',
    'VIEW', 'VIEWS', 'VILL', 'VILLAG', 'VILLAGE', 'VILLAGES', 'VILLE', 'VILLG',
    'VILLIAGE', 'VIS', 'VIST', 'VISTA', 'VL', 'VLG', 'VLGS', 'VLLY', 'VLY',
    'VLYS', 'VST', 'VSTA', 'VW', 'VWS', 'WALK', 'WALKS', 'WALL', 'WAY',
    'WAYS', 'WELL', 'WELLS', 'WL', 'WLS', 'WY', 'XING', 'XRD', 'XRDS'
}

STATE_INITIALS = {
    'AL': 'Alabama',
    'AK': 'Alaska',
    'AZ': 'Arizona',
    'AR': 'Arkansas',
    'CA': 'California',
    'CO': 'Colorado',
    'CT': 'Connecticut',
    'DE': 'Delaware',
    'FL': 'Florida',
    'GA': 'Georgia',
    'HI': 'Hawaii',
    'ID': 'Idaho',
    'IL': 'Illinois',
    'IN': 'Indiana',
    'IA': 'Iowa',
    'KS': 'Kansas',
    'KY': 'Kentucky',
    'LA': 'Louisiana',
    'ME': 'Maine',
    'MD': 'Maryland',
    'MA': 'Massachusetts',
    'MI': 'Michigan',
    'MN': 'Minnesota',
    'MS': 'Mississippi',
    'MO': 'Missouri',
    'MT': 'Montana',
    'NE': 'Nebraska',
    'NV': 'Nevada',
    'NH': 'New Hampshire',
    'NJ': 'New Jersey',
    'NM': 'New Mexico',
    'NY': 'New York',
    'NC': 'North Carolina',
    'ND': 'North Dakota',
    'OH': 'Ohio',
    'OK': 'Oklahoma',
    'OR': 'Oregon',
    'PA': 'Pennsylvania',
    'RI': 'Rhode Island',
    'SC': 'South Carolina',
    'SD': 'South Dakota',
    'TN': 'Tennessee',
    'TX': 'Texas',
    'UT': 'Utah',
    'VT': 'Vermont',
    'VA': 'Virginia',
    'WA': 'Washington',
    'WV': 'West Virginia',
    'WI': 'Wisconsin',
    'WY': 'Wyoming'
}

class CSV_SCRAPY(csv.Dialect):
    delimiter = "|"
    quotechar = "'"
    doublequote = True
    skipinitialspace = False
    lineterminator = "\n"
    quoting = csv.QUOTE_MINIMAL

class StreetAddress(collections.namedtuple('StreetAddress', 'house_number street city state zip_code raw')):
    LINE1_REGEX = re.compile(r'(?P<house_number>[0-9]{1,10}) (?P<street>([A-Z][a-z]* ?)+? (?P<street_suffix>[A-Z][a-z]{1,9}))\.?( [NSEW]{1,2})?')
    LINE2_REGEX = re.compile(r'(?P<city>([A-Z][a-z]+ ?)+?),? (?P<state>[A-Z]{2})( (?P<zip_code>[0-9]{5}(-[0-9]{4})?))?')

    NLP_TOKENIZER = nltk.RegexpTokenizer(r'''(?x)
         https?://[^ ]+  # URLs
       | \@[0-9A-Za-z_]+ # Twitter usernames
       | \#[0-9A-Za-z_]+ # Twitter hashtags
       | \w+             # Words/punctuation
       | [^\w\s]+
    ''')

    NLP_PARSER = nltk.RegexpParser(r'''
    NP: {<(NNP.?|CD)>+}
        {<DT|PP\$>?<(JJ|CD)>*<(NN[A-Z]?|CD)>+}
    ''')

    def __str__(self):
        addrstr = ""

        if self.street is not None:
            addrstr += self.house_number + " " + self.street

            if self.city is not None:
                addrstr += ", "

        if self.city is not None:
            addrstr += self.city + ", " + self.state

            if self.zip_code is not None:
                addrstr += " " + self.zip_code

        return addrstr

    def dict(self):
        ret = {"country": "USA"}

        if self.street is not None:
            ret["street"] = self.house_number + " " + self.street

        if self.city is not None:
            ret["city"] = self.city
            ret["state"] = self.state

            if self.zip_code is not None:
                ret["postalcode"] = self.zip_code

        return ret

    @staticmethod
    def nlp(text):
        '''

        '''

        word_forest = [StreetAddress.NLP_PARSER.parse(nltk.pos_tag(StreetAddress.NLP_TOKENIZER.tokenize(sentence))) for sentence in nltk.sent_tokenize(text)]
        word_forest.reverse()

        while word_forest:
            tree = word_forest.pop()

            if type(tree) is nltk.Tree:
                if tree.label() == "NP" and tree[0][0].isdigit():
                    for i in range(2, len(tree)):
                        if tree[i][0].upper() in STREET_SUFFIXES:
                            leaves = [j[0] for j in tree.leaves()]

                            return StreetAddress(
                                house_number = tree[0][0],
                                street = " ".join(leaves[1:(i + 1)]),
                                city = None,
                                state = None,
                                zip_code = None,
                                raw = " ".join(leaves)
                            )
                else:
                    for subtree in tree[::-1]:
                        word_forest.append(subtree)

        return None

    @staticmethod
    def re(text):
        '''

        '''

        line1 = StreetAddress.LINE1_REGEX.search(text)
        line2 = StreetAddress.LINE2_REGEX.search(text)

        house_number = None
        street       = None
        city         = None
        state        = None
        zip_code     = None
        raw          = ""

        if line1 is not None:
            raw = line1.string[line1.start():line1.end()]

            if line1.group('street_suffix').upper() in STREET_SUFFIXES:
                house_number  = line1.group('house_number')
                street        = line1.group('street')
            else:
                line1 = None

        if line2 is not None:
            if raw:
                raw += " "

            raw += line2.string[line2.start():line2.end()]

            if line2.group('state') in STATE_INITIALS:
                city     = line2.group('city')
                state    = line2.group('state')
                zip_code = line2.group('zip_code')
            else:
                line2 = None

        if line1 is None and line2 is None:
            return None

        return StreetAddress(
            house_number = house_number,
            street = street,
            city = city,
            state = state,
            zip_code = zip_code,
            raw = raw
        )

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
            log = sys.stderr#open(LOGNAME, "a")
            ifd = open(sys.argv[1], "r", newline = '') if len(sys.argv) > 1 else sys.stdin
            ofd = open(sys.argv[2], "w", newline = '') if len(sys.argv) > 2 else sys.stdout

            process_csv(ifd, ofd, log)
#        except Exception as err:
#            log.write("%s -> %s: Uncaught exception of type %s\n" % (ifd.name, ofd.name, str(type(err))))

#            for line in traceback.format_exc().splitlines():
#                log.write("%s -> %s: %s\n" % (ifd.name, ofd.name, line))

#            log.flush()
        finally:
            #log.close()
            ifd.close()
            ofd.close()

GEODB.close()
AREADB.close()
