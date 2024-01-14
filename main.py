import pandas as pd
import os
from langchain_community.document_loaders import DataFrameLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

def load_data(file_name: str = "text_data.csv"):
    """
    convert the csv into pandas data frame format

    Parameters
    ----------
    file_name: str

    Return
    ------
    df: pd.DataFrmae
    """
    data_root = 'data/'
    file_path = os.path.join(data_root, file_name)
    df = pd.read_csv(file_path)
    return df



if __name__ == "__main__":
    df = load_data("text_data.csv")
    print(df.columns)
