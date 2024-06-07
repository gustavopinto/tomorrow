from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import os
load_dotenv()

model = ChatOpenAI(openai_api_key=os.getenv('OPENAI_API_KEY'))

prompt = ChatPromptTemplate.from_messages([
    ("system", "Quero que você atue como um cozinheiro especializado em comida Paraense. Responda em no maximo 30 palavaras."),
    ("user", "{user_query}"), # a gente que decide os nomes e as quantidades de parametros do prompt
    ("system", "Para responder, considere o histórico recente de conversa: `{history}`."),
])


## esse é o minimo que a gente precisa para fazer uma requisicao ao modelo
## 1. modelo
## 2. prompt 
## esse operador `|` (pipe) combina o modelo e prompt, gerando que se chama de `chain`
chain = prompt | model 

response = chain.invoke({
    "user_query" : "Oi, meu nome é Gustavo e eu tomo tacaca todos os dias.",
    "history" : None
})

print(response.content)