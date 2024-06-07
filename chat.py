from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import os
load_dotenv()


ChatOpenAI(openai_api_key=os.getenv('OPENAI_API_KEY'))