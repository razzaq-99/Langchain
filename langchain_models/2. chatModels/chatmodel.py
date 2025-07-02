from langchain_ollama import OllamaChatModel
from langchain_groq import GrokChatModel
from langchain_core.messages import HumanMessage
from dotenv import load_dotenv

load_dotenv()

model = OllamaChatModel(model="gemma:2b")
# model = GrokChatModel(
#     model_name="llama3-70b-8192"  
# )

response = model.invoke("What is the capital of Pakistan?",)

print(response)
