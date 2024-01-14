"""
web scrab a website for adding additional context
-------------------------------------------------
"""

import time
import pandas as pd 
from tqdm import tqdm
from trafilatura.sitemaps import sitemap_search
from trafilatura import fetch_url, extract, extract_metadata

