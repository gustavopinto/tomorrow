from langchain_community.document_loaders import PyPDFLoader
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from dotenv import load_dotenv
load_dotenv()

loader = PyPDFLoader("https://arxiv.org/pdf/2210.07342")
pages = loader.load() # devolve um Document, envelopando a string
pages =  [page.page_content for page in pages] # tiro a string de dentro do document

text_splitter = RecursiveCharacterTextSplitter(
    separators=[
        "\n\n",
        "\n",
        ".",
        "!",
        "?",
        ";",
        " ",
        "",
    ],
    chunk_size=1000,
    chunk_overlap=200,
    length_function=len,
    is_separator_regex=False,
)

chunks = text_splitter.split_text("\n".join(pages)) # chunks estavam em str

docs = [Document(page_content=chunk) for chunk in chunks] # chunks estao em formato de Document

db = Chroma.from_documents(docs, 
                           OpenAIEmbeddings(), 
                           persist_directory="./tomorrow")
