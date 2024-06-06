from langchain_community.llms import HuggingFaceEndpoint
import os

os.environ["HUGGINGFACEHUB_API_TOKEN"] = "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"

model = HuggingFaceEndpoint(repo_id="meta-llama/Meta-Llama-3-8B-Instruct", task="text-generation")


while True:
    user_message = input("> ")

    prompt = f"""
    <|begin_of_text|>

    <|start_header_id|>system<|end_header_id|>
    Quero que você atue como um cozinheiro especializado em comida Paraense. 
    Responda em no maximo 30 palavaras.
    <|eot_id|>

    <|start_header_id|>user<|end_header_id|>
    {user_message}
    <|eot_id|>

    <|start_header_id|>assistant<|end_header_id|>
    """

    response = model.invoke(prompt)
    print(f"> {response}")
    print("\n")

