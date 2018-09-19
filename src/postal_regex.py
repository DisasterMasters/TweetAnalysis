import collections
import re
import sys

#import tweepy
from geopy.geocoders import Nominatim

# Obtained from <https://pe.usps.com/text/pub28/28apc_002.htm>
STREET_SUFFIXES = frozenset([
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
])

STATE_INITIALS = frozenset([
    'AL', 'AK', 'AZ', 'AR', 'CA', 'CO', 'CT', 'DE', 'FL', 'GA', 'HI', 'ID',
    'IL', 'IN', 'IA', 'KS', 'KY', 'LA', 'ME', 'MD', 'MA', 'MI', 'MN', 'MS',
    'MO', 'MT', 'NE', 'NV', 'NH', 'NJ', 'NM', 'NY', 'NC', 'ND', 'OH', 'OK',
    'OR', 'PA', 'RI', 'SC', 'SD', 'TN', 'TX', 'UT', 'VT', 'VA', 'WA', 'WV',
    'WI', 'WY', 'AS', 'DC', 'FM', 'GU', 'MH', 'MP', 'PW', 'PR', 'VI'
])

LINE1REGEX = re.compile(r'(?P<house_number>[0-9]{1,10}) (?P<street>([A-Z][a-z]* ?)+? (?P<street_suffix>[A-Z][a-z]{1,9}))')
LINE2REGEX = re.compile(r'(?P<city>([A-Z][a-z]+ ?)+?),? (?P<state>[A-Z]{2})( (?P<zip_code>[0-9]{5}(-[0-9]{4})?))?')

class StreetAddress(collections.namedtuple('StreetAddress', 'house_number street city state zip_code')):
    @staticmethod
    def match(text):
        line1 = LINE1REGEX.search(text)
        line2 = LINE2REGEX.search(text)

        house_number = None
        street       = None
        city         = None
        state        = None
        zip_code     = None

        if line1 is not None:
            if line1.group('street_suffix').upper() in STREET_SUFFIXES:
                house_number  = line1.group('house_number')
                street        = line1.group('street')
            else:
                line1 = None

        if line2 is not None:
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
            zip_code = zip_code
        )

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

    def geocode(self, geolocator):
        results = geolocator.geocode(str(self), exactly_one = False)

        if results is not None:
            for coord in results:
                if "USA" in coord.address:
                    return (coord.latitude, coord.longitude)

        return None

if len(sys.argv) < 2:
    print("Usage: " + sys.argv[0] + " [input_file]")
    exit(1)

all_tweets = []
tweets_w_addrs = []
tweets_w_coords = []

# Use OpenStreetMap because it's free, might switch to Google Maps later
geolocator = Nominatim(user_agent = "disaster-masters-curent-utk")

with open(sys.argv[1]) as fd:
    for tweet in fd.read().split('\t'):
        tweet = tweet.split('~+&$!sep779++')

        if tweet == ['']:
            continue

        all_tweets.append(tweet)

        text = tweet[0]
        link = tweet[2]

        addr = StreetAddress.match(text)

        if addr is not None:
            tweet = tweet.copy()
            tweet.append(addr)

            tweets_w_addrs.append(tweet)

            coords = addr.geocode(geolocator)

            if coords is not None:
                tweet = tweet.copy()
                tweet.append(coords)

                tweets_w_coords.append(tweet)

for i in tweets_w_coords:
    print("\"" + i[0] + "\" " + str(i[-2]) + ", " + str(i[-1]))

'''
Don't do anything yet with the data, except print the percentage of tweets with
potential address information to stdout. We are going to need access to the
Twitter API (which needs to be approved) before we can do anything else.
'''
print("Total tweets                                : %d" % len(all_tweets))
print("---------------------------------------------")
print("Tweets with an address in their text        : %d" % len(tweets_w_addrs))
print("Percentage                                  : %f %%" % (100 * len(tweets_w_addrs) / len(all_tweets)))
print("---------------------------------------------")
print("Tweets whose address is a valid geolocation : %d" % len(tweets_w_coords))
print("Percentage                                  : %f %%" % (100 * len(tweets_w_coords) / len(all_tweets)))
