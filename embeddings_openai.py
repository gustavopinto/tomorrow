from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv
import os
load_dotenv()

embeddings = OpenAIEmbeddings(api_key=os.getenv('OPENAI_API_KEY'))
hw = embeddings.embed_query("hello world")
print(hw)