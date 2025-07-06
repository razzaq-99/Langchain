from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnableLambda,RunnableBranch
from dotenv import load_dotenv
from langchain_ollama import ChatOllama
from langchain_core.output_parsers import StrOutputParser 
import os

load_dotenv()

os.environ["LANGCHAIN_TRACING_V2"] = "false"
os.environ["LANGCHAIN_API_KEY"] = "" 
model = ChatOllama(model="gemma:2b")


positive_feedback = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant"),
        ("human","Generate a thank you note for this positive feedback: {feedback}")
    ]
)

negative_feedback = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant"),
        ("human","Generate a polite note for this negative feedback: {feedback}")
    ]
)

neutral_feedback = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant"),
        ("human","Generate a request for more details for this neutral feedback: {feedback}")
    ]
)

escalate_feedback_template = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant."),
        (
            "human",
            "Generate a message to escalate this feedback to a human agent: {feedback}.",
        ),
    ]
)


classification_template = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a feedback classifier."),
        (
            "human",
            "Classify this feedback as positive, negative, neutral or escalate: {feedback}",
        ),
    ]
)


branches = RunnableBranch(
    (
        lambda x: "positive" in x,
        positive_feedback | model | StrOutputParser()
    ),
    (
        lambda x: "negative" in x,
        negative_feedback | model | StrOutputParser()
    ),
    (
        lambda x: "neutral" in x,
        neutral_feedback | model | StrOutputParser()
    ),
    
    escalate_feedback_template | model | StrOutputParser()
)


classification_chain = (
    classification_template
    | model
    | StrOutputParser()
)


chain = classification_chain | branches

# result = chain.invoke({"feedback": "I love the new features of your product!"})
# print(f"Response:\n{result}")

# result = chain.invoke({"feedback": "The product is okay, but it could be better."})
# print(f"Response:\n{result}")

# result = chain.invoke({"feedback": "I am not satisfied with the product, it has many issues."})
# print(f"Response:\n{result}")

result = chain.invoke({"feedback": "I have some concerns about the product, can you help?"})
print(f"Response:\n{result}")