from langchain_community.chat_models import ChatOpenAI
from langchain_community.llms import OpenAI
import os
from langchain_community.embeddings import OpenAIEmbeddings


llm = OpenAI(api_key=os.environ['OPENAI_API_KEY'], temperature=0.7, max_tokens= 2048)
vision_model = ChatOpenAI(model="gpt-4-vision-preview", max_tokens=2048)
embedding_model = OpenAIEmbeddings(api_key=os.environ['OPENAI_API_KEY'], model="text-embedding-ada-002")