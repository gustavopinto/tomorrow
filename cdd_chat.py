from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from dotenv import load_dotenv
import os
load_dotenv()

def get_cdd_info(pdf):
    """vai me devolver uma lista de chunks em str"""

    loader = PyPDFLoader(pdf)
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

    chunks = text_splitter.split_text("\n".join(pages))
    return chunks


# 0. preciso experimentar com o tamanho adequado de chunks: https://huggingface.co/spaces/m-ric/chunk_visualizer
# 1. preciso separar os dados do ruído
# 2. tiraria as referencias
# 3. corrigiria os hifens do textos
# 4. removeria (ou pediria para outro modelo explicar) as tabelas
# 5. sci-fiction (https://dl.acm.org/doi/pdf/10.1145/3616855.3635752)

import sys; sys.exit(0)

model = ChatOpenAI(openai_api_key=os.getenv('OPENAI_API_KEY'), model="gpt-3.5")

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