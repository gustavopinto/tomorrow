# from langchain_openai import OpenAIEmbeddings
# from dotenv import load_dotenv
# import os
# load_dotenv()

# embeddings = OpenAIEmbeddings(api_key=os.getenv('OPENAI_API_KEY'))
# hw = embeddings.embed_query("hello world")
# print(hw)


from sentence_transformers import SentenceTransformer

# 1. Load a pretrained Sentence Transformer model
model = SentenceTransformer("all-MiniLM-L6-v2")

# The sentences to encode
sentences = ["hello world"]

embeddings = model.encode(sentences)
print(embeddings)