import os
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import BedrockEmbeddings
from langchain.vectorstores import FAISS
from langchain.indexes import VectorstoreIndexCreator
import boto3
from langchain_community.llms.bedrock import Bedrock


ACCESS_KEY = "your access key"
SECRET_ACCESS_KEY = "your secret access key"

bedrock_client = boto3.client('bedrock-runtime',
                              aws_access_key_id=ACCESS_KEY,
                              aws_secret_access_key=SECRET_ACCESS_KEY,
                              region_name="us-east-1")


def get_index():
    # ## Load the document into the memory
    data_load = PyPDFLoader("https://www.upl-ltd.com/images/people/downloads/Leave-Policy-India.pdf")

    # ## Data Transform 
    data_split = RecursiveCharacterTextSplitter(separators=["\n\n","\n"," ",""],chunk_size=100,chunk_overlap=10)

    # ## Embedding 
    data_embedding = BedrockEmbeddings(
        client=bedrock_client,
        model_id="amazon.titan-embed-text-v2:0",
    )
    # creating vector index
    data_index = VectorstoreIndexCreator(
        text_splitter=data_split,
        embedding=data_embedding,
        vectorstore_cls=FAISS
    )

    db_index = data_index.from_loaders([data_load])
    return db_index

def get_llm():
    llm = Bedrock(
            client=bedrock_client,
            model_id="meta.llama3-70b-instruct-v1:0",
            model_kwargs={
                "temperature":0.1,
                "top_p":0.9
            })
    return llm



def rag_response(index,question):
    rag_query = index.query(question=question,llm=get_llm())
    return rag_query


