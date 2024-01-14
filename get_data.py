"""
web scrab a website for adding additional context
-------------------------------------------------
"""

import time
import pandas as pd 
from tqdm import tqdm
from trafilatura.sitemaps import sitemap_search
from trafilatura import fetch_url, extract, extract_metadata

def get_urls_from_sitemap(source_url: str) -> list:
    """
    from the given source url link get all the url for a soure web page
    and return list of urls

    Parameters
    ----------
    source_url: str


    Return
    ------
    urls: list 
    """
    urls = sitemap_search(source_url)
    return urls

def create_data(websites: list) -> pd.DataFrame:
    """
    given the list of websites, extract the knowledge from the each website along with
    the meta data and store it into a Pandas Data Frame

    Parameters
    ----------
    websites: list

    Return
    ------
    df: pd.DataFrame
    """
    data = []
    for website in tqdm(websites, desc= "Websites"):
        urls = get_urls_from_sitemap(website)
        for url in tqdm(urls, desc = "URLs"):
            html_content = fetch_url(url)
            body = extract(html_content)
            try:
                metadata = extract_metadata(html_content)
                title = metadata.title
                description = metadata.description
            except:
                metadata = ""
                title = ""
                description = ""
            d_info = {
                 "url": url,
                 "body": body,
                 "title": title,
                 "description": description
            }
            data.append(d_info)
            time.sleep(0.45)
    df = pd.DataFrame(data)
    df = df.drop_duplicates()
    df = df.dropna()
    return df

if __name__ == "__main__":
    source_url = ["https://python.langchain.com/"]
    data_frame = create_data(source_url)
    data_frame.to_csv("data/text_data.csv", index= False)
