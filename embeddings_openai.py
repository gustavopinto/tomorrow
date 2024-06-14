from langchain_openai import OpenAIEmbeddings   
from dotenv import load_dotenv
import numpy as np
import os
load_dotenv()

embeddings = OpenAIEmbeddings(api_key=os.getenv('OPENAI_API_KEY'), model="text-embedding-3-large")

user_query = "what kind of food I eat in Italy?"
user_query_emb = embeddings.embed_query(user_query) # somente uma string


chunks = ["pizza", # imagine que essa lista é um banco de dados
             "man", 
             "king", 
             "google", 
             "federal university of bahia", # ruido
             "lasagna"]


chunks_emb = embeddings.embed_documents(chunks) # passo uma lista

for i in range(len(chunks_emb)):
    similarities = np.dot(chunks_emb[i], user_query_emb)
    print(chunks[i], user_query, similarities)
