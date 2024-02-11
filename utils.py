"""
The utility functions 
=====================

development of a chatbot using RAG (reterival augmented generation)
"""
import os
import hydra
from omegaconf import OmgeaConf, DictConfig


def load_text_file(file: str):
    """
    Load the text file containing the useful context information 

    Parameters
    ----------
    file: (a txt file containing knowledge)
    """
    with open(file, 'r') as f:
        context = f.read()
    return context 

hydra.main(config_name = "configs", config_path = 'conf', base_version = None)
def get_data(cfg: DictConfig):
    dataset = cfg.data.load_text_file
    if os.path.exist(dataset):
        print(dataset)
    


if __name__ == "__main__":
    pass