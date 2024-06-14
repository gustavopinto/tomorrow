from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction
import chromadb
import os

def select_similar_chunks(user_query):
    client = chromadb.PersistentClient(path="./tomorrow")
    col = client.get_or_create_collection("langchain", 
                                        embedding_function=OpenAIEmbeddingFunction(api_key=os.getenv('OPENAI_API_KEY')))
    results =  col.query(query_texts=[user_query], n_results=10)
    return "\n".join(results['documents'][0])

def prompt_llm(user_query, chunks):
    model = ChatOpenAI(openai_api_key=os.getenv('OPENAI_API_KEY'))
    prompt = ChatPromptTemplate.from_messages([
        ("system", "Quero que você atue como um expert em técnicas de design de código."),
        ("system", "Um exemplo de técnica de design de código é a refatoracao, que promove a reestruturação do código."),
        ("user", "{user_query}"),
        #("system", "Se o usuário perguntar algo que não seja sobre design de código, se limite a responder 'Eu não sei'."),
        ("system", "As descricoes sobre CDD estao a seguir: {cdd_info}. Se limite a responder com base nessas informacoes fornecidas. Tente nao trazer outras informacoes na sua resposta."),
        ("system", "Responda em até 150 palavras."),
    ])

    chain = prompt | model 

    response = chain.invoke({
        "user_query" : user_query,
        "cdd_info" : chunks
    })
    return response.content


user_query = "What does CDD mean?"
chunks = select_similar_chunks(user_query)
response = prompt_llm(user_query, chunks)

print(response) 