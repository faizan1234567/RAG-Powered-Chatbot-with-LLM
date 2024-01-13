# RAG-Powered-Chatbot-with-LLM
Pretrained LLM stores factual information in their parameters and, when fine-tuned, provides state-of-the-art performance on downstream tasks. However, their ability to access and provide domain-specific and current knowledge is still limited. To address this, ```Retrieval augmented generation``` (RAG) has been proposed. ```RAG``` uses non-parametric memory to provide additional context to the LLM. It converts the user query to embedding and by using similarity search algorithms it returns ```top K``` best retreived results based on the query. This additional context along with the query is added to the LLM prompt for up-to-date response generation ```[1]```. The following figure shows the RAG working:

![alt text](https://github.com/faizan1234567/RAG-Powered-Chatbot-with-LLM/images/RAG_architecture.png)

## Installation
1. Clone the github repository
```bash
git clone https://github.com/faizan1234567/RAG-Powered-Chatbot-with-LLM
cd RAG-Powered-Chatbot-with-LLM
```
2. create and activate an enviroment using anaconda
```bash
conda create -n rag python=3.10.0 -y
conda activate rag
conda list

python -m pip install --upgrade pip (optional)
pip list
```
3. install required packages
```bash
pip install -r requirements.txt
```
if you face any issue in installation, please create an issue. 

## Acknowledgements
[1]. Lewis, Patrick, et al. "Retrieval-augmented generation for knowledge-intensive nlp tasks." Advances in Neural Information Processing Systems 33 (2020): 9459-9474.

[2]. D’Agostino, A. (2023, November 30). Create a chatbot in python with Langchain and rag. Medium. https://medium.com/mlearning-ai/create-a-chatbot-in-python-with-langchain-and-rag-85bfba8c62d2 