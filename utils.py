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

def create_embeddings(text, model):
    """
    Load the sentence embedding from HuggingFace


    Parameters
    ----------
    text: str
    model_name: name of the encoder
    """
    embeddings = model.encode(text)
    return embeddings

# encode query
def run_query(query, index, model):
  """
  run similarity search
  ---------------------

  parameter:
  query: str a question posed by a user
  index: name of the pincone index
  model: an embedding model
  """
  embedding = model.encode(query).tolist()
  results = index.query(top_k=5, vector=embedding, include_metadata=True, include_values=False)
  return results

# vector database function
def create_vecdb(pdf_doc: str, index, model, chunk_size: int = 500, 
                 chunk_overlap: int = 50, batch_size: int = 4,
                 device: str = 'cpu'):
    
    """
    create vector database using pinecone with cosine similarity matirx and
    768 embedding dimension

    parameters
    ----------
    pdf_loc: dataset file
    index: pinecone index to store vector encodings
    model: embedding model
    chunk_size: each splitted chunk size
    chunk_overlap: chunk_overlap
    batch_size: batches of docs to be encoded
    vec_limited: number of vectors limit for pinecone vec database
    device: by default cpu
    """
    
    # update if cuda available
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # create list of chunked documents from the pdf file
    print('INFO: splitting text')
    context = split_text(pdf_file=pdf_doc, chunk_size= chunk_size, 
                         chunk_overlap= chunk_overlap)
    
    for i in tqdm(range(0, len(context), batch_size)):
        i_end = min(i+batch_size, len(context))
        ids = [str(x) for x in range(i, i_end)]
        metadatas = [{'text': text} for text in context[i:i_end]]
        xc = create_embeddings(context[i:i_end], model)
        records = zip(ids, xc, metadatas)
        index.upsert(vectors=records)

def create_prompt(contexts, query):
    '''
    create a prompt for LLM to answer using the context reterived from the reteriver
    and the quesetion posed by a user

    parameters
    ----------
    contexts: list
    query: str
    '''
    prompt_start = (
    "Answer the question based on the context below.\n\n"+
    "Context:\n")

    prompt_end = (
        f"\n\nQuestion: {query}\nAnswer:")

    prompt = (
        prompt_start + "\n\n---\n\n".join(contexts) + 
        prompt_end
    )

    return prompt




# test utils 
if __name__ == "__main__":
  model_name = "all-mpnet-base-v2"
  model_destination = "sentence-transformers" + "/" + model_name
  model = SentenceTransformer(model_destination)
  utils = Utils()
  PINECONE_API_KEY = utils.get_pinecone_api_key()
  pc = Pinecone(api_key=PINECONE_API_KEY)
  index = pc.Index("test-index")
  dataset_file = 'dataset/sines.pdf'
  if os.path.exists(dataset_file):
      create_vecdb('dataset/sines.pdf', index= index, model= model)
  else:
      print('INFO: the file does not exists')
   