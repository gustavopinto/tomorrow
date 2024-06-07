from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_community.document_loaders import PyPDFLoader

from dotenv import load_dotenv
import os
load_dotenv()

def get_cdd_info():
    loader = PyPDFLoader("https://arxiv.org/pdf/2210.07342")
    pages = loader.load() # devolve um Document, envelopando a string
    pages =  [page.page_content for page in pages] # tiro a string de dentro do document

    return "\n".join(pages) # converto a lista de strings para uma unica string

model = ChatOpenAI(openai_api_key=os.getenv('OPENAI_API_KEY'), model="gpt-4o")

prompt = ChatPromptTemplate.from_messages([
    ("system", "Quero que você atue como um expert em técnicas de design de código."),
    ("system", "Um exemplo de técnica de design de código é a refatoracao, que promove a reestruturação do código."),
    ("user", "{user_query}"),
    #("system", "Se o usuário perguntar algo que não seja sobre design de código, se limite a responder 'Eu não sei'."),
    ("system", "As descricoes sobre CDD estao neste link: {cdd_info}"),
    ("system", "Responda em até 30 palavras."),
])

chain = prompt | model 

response = chain.invoke({
    "user_query" : "Explique quais os principais benefícios do CDD.",
    "cdd_info" : get_cdd_info()
})

print(response.content)