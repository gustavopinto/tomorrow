from langchain_community.llms import HuggingFaceEndpoint
import os
from dotenv import load_dotenv

load_dotenv()

os.environ["HUGGINGFACEHUB_API_TOKEN"] = os.getenv("HG")

model = HuggingFaceEndpoint(repo_id="microsoft/Phi-3-mini-128k-instruct", task="text-generation", temperature=0.9)

history = []

while True:
    user_message = input("> ")
    history.append(user_message)

    prompt = f"""
    <|begin_of_text|>

    <|start_header_id|>system<|end_header_id|>
    Quero que você atue como um cozinheiro especializado em comida Paraense. 
    Responda em no maximo 30 palavaras.
    <|eot_id|>

    <|start_header_id|>user<|end_header_id|>
    {user_message}
    <|eot_id|>

    <|start_header_id|>system<|end_header_id|>
    Para responder, considere o histórico recente de conversa: `{history}`.
    <|eot_id|>

    <|start_header_id|>assistant<|end_header_id|>
    """

    response = model.invoke(prompt)
    history.append(response)

    print(f"> {response}")
    print("\n")