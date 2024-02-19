import os
import hydra
from omegaconf import OmegaConf, DictConfig
from databases.DLAIUtils import Utils
import torch
from pinecone import Pinecone
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
import databases.DLAIUtils as du
import time
import warnings
from tqdm.auto import tqdm
# ignore warnings
from utils import create_vecdb, run_query
warnings.filterwarnings('ignore')

import argparse
import logging

# setup logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

file_handler = logging.FileHandler(filename= "logs/logger.log")
stream_handler = logging.StreamHandler()
formatter = logging.Formatter(fmt= "%(asctime)s: %(message)s", datefmt= '%Y-%m-%d %H:%M:%S')
file_handler.setFormatter(formatter)
stream_handler.setFormatter(formatter)

logger.addHandler(file_handler)
logger.addHandler(stream_handler)

# read command line arguments
def read_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--chunk_size', default= 500, type = int, help= "document chunk size")
    parser.add_argument('--chunk_overlap', default= 50, type = int, help= 'document overlap')
    parser.add_argument('--vecdb', action= "store_true", help= "flag to create vector database")
    parser.add_argument('--batch', type = int, default = 4, help = 'vecdb batch size')
    parser.add_argument('--device', default= 'cpu', type = str, help='cuda or cpu?')
    parser.add_argument('--pdf', type = str, default= "dataset/sines.pdf", help = 'context path')
    parser.add_argument('--print_search', action= "store_true", help="show respone to a query")
    opt = parser.parse_args()
    return opt

@hydra.main(config_name = "configs", config_path = 'conf', version_base= None)
def main(cfg: DictConfig):
    args = read_args()
    
    # initialize document embedding model for encoding query and documents
    logger.info('Loading Embedding model')
    model_name = cfg.embedding.model_name
    model_destination = "sentence-transformers" + "/" + model_name
    model = SentenceTransformer(model_destination)
    
    logger.info('Setting PineCone vector database for document embedding')
    utils = Utils()
    PINECONE_API_KEY = utils.get_pinecone_api_key()
    pc = Pinecone(api_key=PINECONE_API_KEY)
    index = pc.Index("test-index")

    # create vector database if doesn't exists already
    if args.vecdb:
        logger.info("Upserting document embedding to vector database")
        create_vecdb(pdf_doc= cfg.data.pdf_file, index= index, 
                     model= model, chunk_overlap= cfg.reteriver.chunk_overlap,
                     chunk_size= cfg.reteriver.chunk_size)
        logger.info("Successfully created vector database")
    
    # semantic search
    user_query = ""
    results = run_query(query= user_query, index= index, 
                       model = model)
    # show similar results
    if args.print_search:
        for result in results['matches']:
            print(f"{round(result['score'], 2)}: {result['metadata']['text']}")


# run application
if __name__ == "__main__":
    main()
