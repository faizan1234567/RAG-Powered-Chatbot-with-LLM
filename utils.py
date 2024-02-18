"""
The utility functions 
=====================

development of a chatbot using RAG (reterival augmented generation)
"""
import os
import hydra
from omegaconf import OmegaConf, DictConfig
from databases.DLAIUtils import Utils
import torch
from pinecone import Pinecone, ServerlessSpec
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
import databases.DLAIUtils as du
import time
import warnings
from tqdm.auto import tqdm

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
# ignore warnings
warnings.filterwarnings('ignore')

# enable cuda if available
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# load the txt file 
def load_text_file(file: str):
    """
    Load the text file containing the useful context information 

    Parameters
    ----------
    file: (a txt file containing knowledge)
    """
    with open(file, 'r', encoding= 'utf8') as f:
        context = f.read()
    return context 

def split_text(pdf_file: str, chunk_size: int = 500, 
               chunk_overlap: int = 50):
    """
    split the text into small chunks for creating embeddings
    --------------------------------------------------------


    Parameters
    ----------
    pdf_file: a pdf file that provides the context to ingest in the prompt
    chunk_size: text chunk_size to be used for splitting the text
    chunk_overlap: how of the text between two consectuive docs should overalp
    ideally it's less than 100

    Return
    ------
    context: list (list of chunked text ready to be used for embbedings)
    """
    # load the pdf file and split into a docs
    loader = PyPDFLoader(pdf_file)
    docs = loader.load_and_split()

    # Chunk text
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size,
                                                   chunk_overlap=chunk_overlap)
    chunked_documents = text_splitter.split_documents(docs)

    # store the context in the list
    context = []
    for doc in chunked_documents:
        page_content = doc.page_content.replace("\n", " ")
        context.append(page_content)
    return context




#TODO: to create a vec database
def create_vecdb(batch_size: int = 4, vec_limit: int = 100000,
                 device: str = 'cuda'):
    pass

# read text file 
@hydra.main(config_name = "configs", config_path = 'conf', version_base= None)
def get_data(cfg: DictConfig):
    dataset = cfg.data.text_file
    print()
    context = load_text_file(dataset)
    print(context[:200])
    


if __name__ == "__main__":
  get_data()
   