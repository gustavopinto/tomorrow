from langchain_community.llms import HuggingFaceEndpoint
import os

os.environ["HUGGINGFACEHUB_API_TOKEN"] = "sua_chave_aqui"

model = HuggingFaceEndpoint(repo_id="meta-llama/Meta-Llama-3-8B-Instruct", task="text-generation")

response = model.invoke("Hello World!")
print(response)
