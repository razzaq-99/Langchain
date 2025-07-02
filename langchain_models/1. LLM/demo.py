from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage


llm = ChatOllama(model="gemma:2b")


response = llm.invoke("What is the capital of Brazil?",)

print(response.content)  