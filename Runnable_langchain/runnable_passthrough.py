from langchain_ollama import ChatOllama
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
from langchain.schema.runnable import RunnableSequence , RunnablePassthrough , RunnableParallel
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

joke_chain = RunnableSequence(prompt1 | model | parser)

exp_chain = RunnableParallel({
    'joke':RunnablePassthrough(joke_chain),
    'explanation':RunnableSequence(prompt2 | model | parser)
})

final_chain = RunnableSequence(joke_chain | exp_chain)

print(final_chain.invoke({"topic":"Democracy"}))


