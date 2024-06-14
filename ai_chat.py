from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction
import chromadb
import os

client = chromadb.PersistentClient(path="./tomorrow")
col = client.get_or_create_collection("langchain", 
                                      embedding_function=OpenAIEmbeddingFunction(api_key=os.getenv('OPENAI_API_KEY')))
results =  col.query(query_texts=["what are the limitations of CDD?"], n_results=2)

for r in results['documents'][0]:
    print(r)
    print("\n\n")