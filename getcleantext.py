import unicodedata
import re

def getcleantext(text):
    # Normalize Unicode
    cleantext = unicodedata.normalize('NFC', text)
    # Remove characters outside BMP (emojis)
    cleantext = "".join(c for c in cleantext if ord(c) <= 0xFFFF)
    # Remove newlines and tabs
    cleantext = cleantext.replace("\n", " ").replace("\t", " ")
    # Remove HTTP(S) link
    cleantext = re.sub(r"https?://\S+", "", cleantext)
    # Remove pic.twitter.com
    cleantext = re.sub(r"pic.twitter.com/\S+", "", cleantext)
    # Remove @handle at the start of the tweet
    cleantext = re.sub(r"\A(@\w+ ?)*", "", cleantext)
    # Remove via @handle
    cleantext = re.sub(r"via @\w+", "", cleantext)
    # Strip whitespace
    cleantext = cleantext.strip()

    return cleantext
