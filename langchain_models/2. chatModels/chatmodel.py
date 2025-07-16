from langchain_community.chat_models.ollama import ChatOllama
# from langchain_groq import GrokChatModel
from langchain_core.messages import HumanMessage,SystemMessage,AIMessage
from dotenv import load_dotenv

load_dotenv()

model = ChatOllama(model="gemma:2b")
# model = GrokChatModel(
#     model_name="llama3-70b-8192"  
# )

# response = model.invoke("What is the capital of Pakistan?",)

# print(response)


messages = [
    SystemMessage(content="Solve the following math problem"),
    HumanMessage(content="What is multiplication of 10 with 15?"),
]

Response = model.invoke(messages)
print(f"Answer from AI: {Response.content}")  

