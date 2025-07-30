from langchain_ollama import ChatOllama
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
from langchain.schema.runnable import RunnableSequence , RunnableBranch , RunnableParallel , RunnablePassthrough , RunnableLambda
import os


load_dotenv()

os.environ["LANGCHAIN_TRACING_V2"] = "false"
os.environ["LANGCHAIN_API_KEY"] = "" 

prompt1 = PromptTemplate(
    template="Write a detailed report on {topic}",
    input_variables=['topic']
)

prompt2 = PromptTemplate(
    template="Summarize the following text \n {text}",
    input_variables=['text']
)

model = ChatOllama(
    model = "gemma:2b"
)

parser = StrOutputParser()

report_chain = RunnableSequence(prompt1 | model | parser)

summary_chain = RunnableBranch(
    (lambda x: len(x.split()) > 200, RunnableSequence(prompt2 | model | parser)),
    RunnablePassthrough()
)

final_chain = RunnableSequence(report_chain, summary_chain)

print(final_chain.invoke({"topic":"Impact of AI on Society"}))