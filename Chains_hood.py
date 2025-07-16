from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnableLambda,RunnableSequence
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
        ("system", "You are a comedian who tells jokes about {topic}"),
        ("human", "Tell me {joke_count} jokes")
    ]
)

# format_prompt = RunnableLambda(lambda x:prompt_message.format_messages(**x))
# invoke_model = RunnableLambda(lambda x: model.invoke(x))
# parse_output = RunnableLambda(lambda x: x.content)

upper_case = RunnableLambda(lambda x: x.upper())
count_words = RunnableLambda(lambda x: f"Word Count: {len(x.split())}\n{x}")

# Chain = RunnableSequence(
    
#         first = format_prompt,
#         middle = [invoke_model],
#         last = parse_output
    
# )

Chain = prompt_message | model | StrOutputParser() | upper_case | count_words

response = Chain.invoke({
    "topic": "programming",
    "joke_count": 3
})

print(f"Jokes: {response}")

