from langchain_ollama import ChatOllama
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
from langchain.schema.runnable import RunnableSequence
import os


load_dotenv()

os.environ["LANGCHAIN_TRACING_V2"] = "false"
os.environ["LANGCHAIN_API_KEY"] = "" 

prompt1 = PromptTemplate(
    template="Write a joke about {topic}",
    input_variables=['topic']
)

prompt2 = PromptTemplate(
    template="Explain the following joke\n {text}",
    input_variables=['text']
)

model = ChatOllama(
    model = "gemma:2b"
)

parser = StrOutputParser()

chain = RunnableSequence(prompt1 | model | parser | prompt2 | model | parser)

result = chain.invoke({"topic":"Democracy"})

print(result)