from langchain_ollama import ChatOllama
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
import os
from langchain_core.runnables import RunnableParallel

load_dotenv()

os.environ["LANGCHAIN_TRACING_V2"] = "false"
os.environ["LANGCHAIN_API_KEY"] = ""


model = ChatOllama(model="gemma:2b")

prompt1 = PromptTemplate(
    template="Generate short and simple notes on the following text \n {text}.",
    input_variables=["text"],
)

prompt2 = PromptTemplate(
    template="Generate a 4 short questions answers from the following text:\n {text}.",
    input_variables=["text"],
)

prompt3 = PromptTemplate(
    template = "Merge the provided notes and quizz into a single document:\n Notes: {notes}\n Quiz: {quiz}",
    input_variables=["notes", "quiz"],  
)

parser = StrOutputParser()

parallel_chains = RunnableParallel(
    {
        "notes": prompt1 | model | parser,
        "quiz": prompt2 | model | parser
    })

chain = parallel_chains | prompt3 | model | parser

text = """Artificial Intelligence (AI) is a rapidly evolving field that focuses on creating machines capable of performing tasks that typically require human intelligence. This includes learning, reasoning, problem-solving, perception, and language understanding. AI technologies are being applied in various sectors such as healthcare, finance, transportation, and entertainment, leading to significant advancements and efficiencies. However, the rise of AI also raises ethical concerns regarding privacy, job displacement, and decision-making transparency. 
To address these issues, it is essential to develop AI systems that are not only efficient but also ethical and transparent. This involves creating guidelines and regulations that ensure AI technologies are used responsibly and for the benefit of society. Collaboration between governments, industries, and researchers is crucial to establish a framework that promotes innovation while safeguarding public interests.
As AI continues to develop, it is crucial to address these challenges while harnessing its potential to improve our lives and society as a whole.

"""
result = chain.invoke({"text":text})

print(result)

chain.get_graph().print_ascii()