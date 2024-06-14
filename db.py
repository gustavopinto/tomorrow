from langchain_community.document_loaders import PyPDFLoader
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
import chromadb
from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction


from dotenv import load_dotenv
import os
load_dotenv()

# loader = PyPDFLoader("https://arxiv.org/pdf/2210.07342")
# pages = loader.load() # devolve um Document, envelopando a string
# pages =  [page.page_content for page in pages] # tiro a string de dentro do document

# text_splitter = RecursiveCharacterTextSplitter(
#     separators=[
#         "\n\n",
#         "\n",
#         ".",
#         "!",
#         "?",
#         ";",
#         " ",
#         "",
#     ],
#     chunk_size=1000,
#     chunk_overlap=200,
#     length_function=len,
#     is_separator_regex=False,
# )

# chunks = text_splitter.split_text("\n".join(pages))

# docs = [Document(page_content=chunk) for chunk in chunks]

# db = Chroma.from_documents(docs, 
#                            OpenAIEmbeddings(), 
#                            persist_directory="./tomorrow")



# query = "What did the president say about Ketanji Brown Jackson"
# docs = db.similarity_search(query)
# print(docs[0].page_content)

embedding_function = OpenAIEmbeddingFunction(api_key=os.getenv('OPENAI_API_KEY'))


client = chromadb.PersistentClient(path="./tomorrow")
col = client.get_or_create_collection("langchain", embedding_function=embedding_function)
results =  col.query(query_texts=["isso é um teste"], n_results=2)

print(results)