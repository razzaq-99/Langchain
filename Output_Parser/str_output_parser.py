from langchain_ollama import ChatOllama
from dotenv import load_dotenv
import os
from langchain_core.prompts import PromptTemplate

load_dotenv()

os.environ["LANGCHAIN_TRACING_V2"] = "false"
os.environ["LANGCHAIN_API_KEY"] = "" 


model = ChatOllama(model="gemma:2b")

temp1 = PromptTemplate(
    template="Write a detailed report on the {topic}.",
    input_variables=["topic"],
)

temp2 = PromptTemplate(
    template="Write a 4 line summary on the following text:\n {text}.",
    input_variables=["text"],
)

prompt1 = temp1.invoke({"topic":"Influence of AI"})

result1 = model.invoke(prompt1)

prompt2 = temp2.invoke({"text":result1.content})

result2 = model.invoke(prompt2)

print(result1.content)
# print(result2.content)
