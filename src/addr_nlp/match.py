import collections
import re

import nltk

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

STATE_MAP = {
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

STATE_INVMAP = {v: k for k, v in STATE_MAP.items()}
STATE_SET = set(STATE_MAP.values())

REGEX_LINE1 = re.compile(r'(?P<house_number>[0-9]{1,10}) (?P<street>([A-Z][a-z]* ?)+? (?P<street_suffix>[A-Z][a-z]{1,9}))\.?( [NSEW]{1,2})?')
REGEX_LINE2 = re.compile(r'(?P<city>([A-Z][a-z]+ ?)+?),? (?P<state>[A-Z]{2})( (?P<zip_code>[0-9]{5}(-[0-9]{4})?))?')

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

class StreetAddress(collections.namedtuple('StreetAddress', 'house_number street city state zip_code raw')):
    def __str__(self):
        addrstr = ""

        if self.house_number is not None:
            addrstr += self.house_number

        if self.street is not None:
            if addrstr:
                addrstr += " "
            addrstr += self.street

        if self.city is not None:
            if addrstr:
                addrstr += ", "
            addrstr += self.city

        if self.state is not None:
            if addrstr:
                addrstr += ", "
            addrstr += self.state

        if self.zip_code is not None:
            if addrstr:
                addrstr += " "
            addrstr += self.zip_code

        return addrstr

'''
    @staticmethod
    def libpostal(text):
        try:
            from postal.parser import parse_address
        except ModuleNotFoundError:
            return None

        result = {k: v for v, k in parse_address(text)}

        return StreetAddress(
            house_number = result.get("house_number", None),
            street = result.get("road", None),
            city = result.get("city", None),
            state = result.get("state", None),
            zip_code = result.get("postcode", None),
            raw = text
        )
'''

    @staticmethod
    def statemap(text):
        words = text.split()

        # Allow for two-word state names
        words += [w0 + " " + w1 for w0, w1 in zip(words, words[1:])]

        for word in words:
            if word.title() in STATE_SET:
                return StreetAddress(
                    house_number = None,
                    street = None,
                    city = None,
                    state = word.title(),
                    zip_code = None,
                    raw = word
                )

        return None

    @staticmethod
    def nlp(text):
        '''

        '''

        word_forest = [NLP_PARSER.parse(nltk.pos_tag(NLP_TOKENIZER.tokenize(sentence))) for sentence in nltk.sent_tokenize(text)]
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

        line1 = REGEX_LINE1.search(text)
        line2 = REGEX_LINE2.search(text)

        house_number = None
        street       = None
        city         = None
        state        = None
        zip_code     = None
        raw          = ""

        if line1 is not None:
            raw = line1.group()

            if line1.group('street_suffix').upper() in STREET_SUFFIXES:
                house_number  = line1.group('house_number')
                street        = line1.group('street')
            else:
                line1 = None

        if line2 is not None:
            if raw:
                raw += " "

            raw += line2.string[line2.start():line2.end()]

            if line2.group('state') in STATE_MAP:
                city     = line2.group('city')
                state    = STATE_MAP[line2.group('state')]
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
