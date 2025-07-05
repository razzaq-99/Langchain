from langchain_ollama import ChatOllama
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import os

load_dotenv()

# import os
os.environ["LANGCHAIN_TRACING_V2"] = "false"
os.environ["LANGCHAIN_API_KEY"] = ""  # Optional: clears any key that might be set

# os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")

model = ChatOllama(model="gemma:2b")

prompt_message = ChatPromptTemplate.from_messages(
    [
        ("system","You are a comedian who tells jokes about {topic}"),
        ("human","Tell me {joke_count} jokes")
    ]
)

chain = prompt_message | model | StrOutputParser()
# chain = prompt_message | model

response = chain.invoke({
    "topic": "programming",
    "joke_count": 3})

# print(f"Jokes: {response.content}")  

print(response)