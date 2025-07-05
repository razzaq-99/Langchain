from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnableLambda,RunnableSequence
from dotenv import load_dotenv
from langchain_ollama import ChatOllama
import os

load_dotenv()

os.environ["LANGCHAIN_TRACING_V2"] = "false"
os.environ["LANGCHAIN_API_KEY"] = "" 
model = ChatOllama(model="gemma:2b")

prompt_message = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a comedian who tells jokes about {topic}"),
        ("human", "Tell me {joke_count} jokes")
    ]
)

format_prompt = RunnableLambda(lambda x:prompt_message.format_messages(**x))
invoke_model = RunnableLambda(lambda x: model.invoke(x))
parse_output = RunnableLambda(lambda x: x.content)

Chain = RunnableSequence(
    
        first = format_prompt,
        middle = [invoke_model],
        last = parse_output
    
)

response = Chain.invoke({
    "topic": "programming",
    "joke_count": 3
})

print(f"Jokes: {response}")

