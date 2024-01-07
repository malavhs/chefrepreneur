from langchain.output_parsers import ResponseSchema, StructuredOutputParser
from langchain.output_parsers import CommaSeparatedListOutputParser
from langchain.chains import SequentialChain, LLMChain, RetrievalQA, RetrievalQAWithSourcesChain
from langchain_community.vectorstores import Pinecone
import pinecone
import tiktoken
import base64
from utils import models, prompts
from config import config
from helpers import pinecone_create

def restaurant_name_generator(cuisine, cost):
    llm = LLMChain(llm=models.llm, prompt=prompts.creative_name_gen_prompt)
    response = llm.run({"cuisine": cuisine, "cost": cost})
    print(response)
    return response

def existing_names_list(cuisine, cost):
    # Find and set Pinecone Index for retreival
    pinecone_create.pinecone_index_create(config.pinecone_env, config.pinecone_index_name)
    index = pinecone.Index(config.pinecone_index_name)
    embedding_model = models.embedding_model 
    vectorstore = Pinecone(index, embedding_model.embed_query, "text")
    # Create retriever
    retriever = vectorstore.as_retriever(search_kwargs={"k": 10})
    query = f"Can you give me {cuisine} restaurants that are {cost} in cost using the information you can find in the above summary?"
    qa = RetrievalQAWithSourcesChain.from_chain_type(
        llm=models.llm, 
        chain_type="stuff",
        retriever=retriever, 
        return_source_documents=False,
        chain_type_kwargs={"prompt": prompts.retrieval_prompt}

    )
    result = qa({"question": query})
    print(result['answer'])
    return result['answer']

if __name__ == '__main__':
    existing_names_list("Italian", "High")
