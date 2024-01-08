
from utils import models, prompts
from config import config
from langchain_community.vectorstores import Pinecone
import pinecone
import os
from langchain.document_loaders.csv_loader import CSVLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

def csv_loader(file_path):
    loader = CSVLoader(file_path=file_path, csv_args={'delimiter': ','})
    data = loader.load()
    return data

def chunk_data(file_path, chunk_size, chunk_overlap):
    data = csv_loader(file_path)
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    chunked_data = text_splitter.split_documents(data)
    return chunked_data

def pinecone_index_create(env, index_name):
    pinecone.init(api_key=os.environ['PINECONE_API_KEY'], environment=env)
    print("finished init")
    if index_name not in pinecone.list_indexes():
        pinecone.create_index(
            index_name,
            dimension=1536,
            metric='cosine'
        )

def store_vectors(data, env, index_name):
    embedding_model = models.embedding_model
    pinecone_index_create(env, index_name)
    print("Pinecone Index created...")
    index = Pinecone.from_documents(data, embedding_model, index_name=index_name)
    print("Data load to Pinecone complete...")
    return index


if __name__ == '__main__':
    dataset = 'datasets/Zomato_Mumbai_Dataset_clean.csv'
    # this is for storing data into pinecone for the first time
    data = chunk_data(dataset, 100, 20)
    print('Chunking done')
    index = store_vectors(data, config.pinecone_env, config.pinecone_index_name)