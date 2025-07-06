from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnableLambda,RunnableParallel
from dotenv import load_dotenv
from langchain_ollama import ChatOllama
from langchain_core.output_parsers import StrOutputParser 
import os

load_dotenv()

os.environ["LANGCHAIN_TRACING_V2"] = "false"
os.environ["LANGCHAIN_API_KEY"] = "" 
model = ChatOllama(model="gemma:2b")


prompt_message = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a expert product reviewer "),
        ("human", "List the main features the product {product_name}")
    ]
)

def analyze_pros(features):
    pros_template = ChatPromptTemplate.from_messages(
        [
             ("system", "You are an expert product reviewer."),
            (
                "human",
                "Given these features: {features}, list the pros of these features.",
            ),
        ]
    )
    return pros_template.format_messages(features=features)

def analyze_cons(features):
    cons_template = ChatPromptTemplate.from_messages(
        [
             ("system", "You are an expert product reviewer."),
            (
                "human",
                "Given these features: {features}, list the cons of these features.",
            ),
        ]
    )
    return cons_template.format_messages(features=features)


def combine_pros_cons(pros,cons):
    return f"Pros: {pros}\nCons: {cons}"


pros_branch_chain = (
    RunnableLambda(lambda x: analyze_pros(x)) | model | StrOutputParser()
)

cons_branch_chain = (
    RunnableLambda(lambda x: analyze_cons(x)) | model | StrOutputParser()
)


chain = (
    prompt_message
    | model
    | StrOutputParser()
    | RunnableParallel(branches={"pros": pros_branch_chain, "cons": cons_branch_chain})
    | RunnableLambda(lambda x: combine_pros_cons(x["branches"]["pros"], x["branches"]["cons"]))
)


result = chain.invoke({"product_name": "iPhone 15 Pro Max"})

print(f"Product Review:\n{result}")