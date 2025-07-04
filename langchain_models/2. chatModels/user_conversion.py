from langchain_community.chat_models.ollama import ChatOllama
# from langchain_groq import GrokChatModel
from langchain_core.messages import HumanMessage,SystemMessage,AIMessage
from dotenv import load_dotenv

load_dotenv()

model = ChatOllama(model="gemma:2b")


chat_history = []

system_message = SystemMessage(content="You are a helpful assistant.")
chat_history.append(system_message)

while True:
    query = input("You: ")
    if query.lower()== "exit":
        break
    chat_history.append(HumanMessage(content=query))
    
    result = model.invoke(chat_history)
    response = result.content
    chat_history.append(AIMessage(content=response))
    
    print(f"AI: {response}")
    
    
    print("---Message History---")
    print(chat_history)