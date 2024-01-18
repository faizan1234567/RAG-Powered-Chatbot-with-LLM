# Developing a Chatbot for SINES using RAG
In this project we utilize RAG to build question answering chatbot for SINES. In this file I am writing instructions to install the packages
and how to run the code in the notebook. The overall project directory structure is shown below:

```bash
    RAG_chatbot_implementation
    ├── dataset
    │   ├── sines.pdf
    │   └── sines.txt
    ├── requirements.txt
    ├── README.md
    └── building_RAG_powered_chatbot_for_SINES_updated.ipynb 
```

## Installation
If we are running the notebook locally, it is a good practice to create a virtual environment using anaconda or python venv.

```bash
# create an virtual enviroment
conda create -n rag_chatbot python=3.10 -y
conda activate rag_chatbot
conda list

# upgrade pip if its out dated
python -m pip install --upgrade pip (optional)

# installation
pip install -r requirements.txt
pip list
```

The next step is to link the notebook with virtual enviroment we just created. 

```bash
# install jupyter notebook if its installted
pip install jupyter

# install ipykernel and link notebook with venv
pip install ipykernel 
python -m ipykernel install --user --name=rag_chatbot

# now we can use notebook with installed packages in rag_chatbot enviroment

jupyter notebook
```
assuming we are at the root of the project folder. 

## Google Colab usage
In the google colab we need to open the notebook and enable Tesla T4 GPU.

Then for installation we need to run
```bash
!pip install -r requirments.txt
```
we can load the data files in content/ directory by uploading files in content (default). Or we can import google drive 
and mount it and store files there.
```python
from google.colab import drive
drive.mount('/gdrive')
```
