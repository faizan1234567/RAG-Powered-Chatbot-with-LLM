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
# ignore warnings
warnings.filterwarnings('ignore')

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

# enable cuda if available
device = 'cuda' if torch.cuda.is_available() else 'cpu'
#TODO: to create a vec database
def create_vecdb(batch_size: int = 200, vec_limit: int = 100000,
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
   