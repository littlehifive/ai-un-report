import os

API_BASE_URL = "https://digitallibrary.un.org/api/v1/search"
BASE_URL = "https://digitallibrary.un.org/search"

MAX_NB_RESULTS = 200

FORMATS = {
    "bibtex": "btex",
    "marc": "hm",
    "marcxml": "xm",
    "dublincore": "xd",
    "endnote": "xe",
    "nlm": "xn",
    "refworks": "xw",
    "ris": "ris",
}

DEFAULT_API_FORMAT = "xml"
DEFAULT_FORMAT = "marcxml"

API_KEY = os.getenv("UN_API")
