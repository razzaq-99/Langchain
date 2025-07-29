from langchain_ollama import ChatOllama
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
from langchain.schema.runnable import RunnableSequence , RunnableLambda , RunnableParallel , RunnablePassthrough
import os


load_dotenv()

os.environ["LANGCHAIN_TRACING_V2"] = "false"
os.environ["LANGCHAIN_API_KEY"] = "" 

prompt1 = PromptTemplate(
    template="Write a joke about {topic}",
    input_variables=['topic']
)

model = ChatOllama(
    model = "gemma:2b"
)

parser = StrOutputParser()

joke_chain = RunnableSequence(prompt1 | model | parser)

parallel_chain = RunnableParallel({
    'joke':RunnablePassthrough(),
    'word_count':RunnableLambda(lambda x: len(x.split())),
})

final_chain = RunnableSequence(joke_chain, parallel_chain)

print(final_chain.invoke({"topic":"Democracy"}))