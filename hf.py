from langchain_community.llms import HuggingFaceEndpoint
import os
from dotenv import load_dotenv

load_dotenv()

os.environ["HUGGINGFACEHUB_API_TOKEN"] = os.getenv("HG")

model = HuggingFaceEndpoint(repo_id="microsoft/Phi-3-mini-4k-instruct", task="text-generation", temperature=0.9)


prompt = f"""
<|user|>
How to explain Internet for a medieval knight?<|end|>
<|assistant|>
"""

response = model.invoke(prompt)
print(f"> {response}")
