from langchain_ollama import ChatOllama
from dotenv import load_dotenv
import os
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

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

parser = StrOutputParser()

chain = temp1 | model | parser | temp2 | model | parser

result = chain.invoke({"topic":"Influence of AI"})

print(result)

