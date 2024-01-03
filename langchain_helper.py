from langchain.llms import OpenAI
from langchain.prompts import ChatPromptTemplate, PromptTemplate
from langchain.output_parsers import ResponseSchema, StructuredOutputParser
from langchain.chains import SequentialChain, LLMChain, RetrievalQA, RetrievalQAWithSourcesChain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders.csv_loader import CSVLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Pinecone
import os
import pinecone
import tiktoken

def pinecone_index_create(env, index_name):
    pinecone.init(api_key=os.environ['PINECONE_API_KEY'], environment=env)
    print("finished init")
    if index_name not in pinecone.list_indexes():
        pinecone.create_index(
            index_name,
            dimension=1536,
            metric='cosine'
        )

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

def store_vectors(data, env, index_name):
    embedding_model = OpenAIEmbeddings(
        api_key=os.environ['OPENAI_API_KEY'],
        model="text-embedding-ada-002"
    )
    pinecone_index_create(env, index_name)
    print("created....")
    index = Pinecone.from_documents(data, embedding_model, index_name=index_name)
    return index

def generate_names(cuisine, cost):
    llm = OpenAI(api_key=os.environ['OPENAI_API_KEY'],temperature=0.7)
    pinecone_index_create('gcp-starter', 'restaurant-project')
    index = pinecone.Index('restaurant-project')
    embedding_model = OpenAIEmbeddings(
        api_key = os.environ['OPENAI_API_KEY'],
        model="text-embedding-ada-002"
    )
    vectorstore = Pinecone(index, embedding_model.embed_query, "text")
    query = f"Tell me {cuisine} restaurants that have {cost} cost associated with them from the documents"
    retriever = vectorstore.as_retriever(search_kwargs={"k": 10})
    QA_PROMPT = PromptTemplate(
        input_variables=["summaries", "question"],
        template="""Act as a person who understands the restaurant naming business. If the question cannot be answered,
    using the information provided then say "I dont know". Remove any duplicates from the final answer. Answer should be in a numbered list format.
    
    {summaries}
    
    Question: {question}
    
    Give it as a list. 
    """
    )
    qa = RetrievalQAWithSourcesChain.from_chain_type(
        llm=llm, chain_type="stuff",
        retriever=retriever, return_source_documents=False,
        chain_type_kwargs={"prompt": QA_PROMPT}

    )
    result = qa({"question": query})
    print(result['answer'])
    return result['answer']

if __name__ == '__main__':
    pinecone_env = 'gcp-starter'
    pinecone_index_name = 'restaurant-project'
    # dataset = 'datasets/Zomato_Mumbai_Dataset_clean.csv'
    # # this is for storing data into pinecone for the first time
    # data = chunk_data(dataset, 100, 20)
    # print('Chunking done')
    # index = store_vectors(data, pinecone_env, pinecone_index_name)
