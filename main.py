import pandas as pd
import os
from langchain_community.document_loaders import DataFrameLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OpenAIEmbeddings, SelfHostedHuggingFaceEmbeddings, HuggingFaceInstructEmbeddings
from dotenv import load_dotenv
#initailize
load_dotenv()

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


def chunk_dataset(dataset: pd.DataFrame, chunk_size: int = 1000, chunk_overlap: int = 0):
    """
    divide the dataset into chunk of documents due to limited context window size of
    an LLM by using langchain functions. 

    Parameters
    ----------
    dataset: a dataframe containing information
    chunk_size: number of elements in each chunk
    chunk_overlap: chunk overlap size

    Return
    ------
    list: list of chunks
    """
    chunks = DataFrameLoader(
           dataset, page_content_column= "body"
    ).load_and_split(text_splitter = 
                     RecursiveCharacterTextSplitter(chunk_size = chunk_size, 
                                                    chunk_overlap = chunk_overlap,
                                                    length_function = len), 
                     )
    
    # for each chunk retreive important info
    for doc in chunks:
        title = doc.metadata["title"]
        description = doc.metadata["description"]
        content = doc.page_content
        url = doc.metadata["url"]
        updated_content = f"TITLE: {title}\DESCRIPTION: {description}\BODY: {content}\nURL: {url}"
        doc.page_content = updated_content
    return chunks
    

def create_or_retreive_store(chunks: list):
    """
    create a embeddings and store them using FAISS
    create an embedding of each document chunk

    Parameters
    ----------
    chunks: list of docs chunks

    Return
    ------
    FAISS: vector store for storing vector embeddings
    """
    # embeddings = OpenAIEmbeddings()
    embeddings = HuggingFaceInstructEmbeddings() 
    if not os.path.exists("/db"):
        print('Creating embeddings')
        vectorstore = FAISS.from_documents(
            chunks,  
        )
        vectorstore.save_local("/db")
    else:
        print("Loading DB")
        vectorstore = FAISS.load_local("/db", embeddings)
    return vectorstore





if __name__ == "__main__":
    df = load_data("text_data.csv")
    text_chunks = chunk_dataset(df)
    # test a chunk
    # print(text_chunks[74])
    vectorstore = create_or_retreive_store(text_chunks)
    print(type(vectorstore))
