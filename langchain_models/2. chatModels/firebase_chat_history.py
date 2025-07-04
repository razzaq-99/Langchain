from langchain_community.chat_models.ollama import ChatOllama
# from langchain_groq import GrokChatModel
from langchain_core.messages import HumanMessage,SystemMessage,AIMessage
from dotenv import load_dotenv
from google.cloud import firestore
from langchain_google_firestore import FirestoreChatMessageHistory 



load_dotenv()

PROJECT_ID = "your-project-id"  
SESSION_ID = "your-session-id"  
COLLECTION_NAME = "your-collection-name"

print("Initializing Chat History...")

chat_history = FirestoreChatMessageHistory(
    project_id=PROJECT_ID,
    session_id=SESSION_ID,
    collection_name=COLLECTION_NAME
)

print("Chat History Initialized.")
print("Current chat history:", chat_history.messages)

model = ChatOllama(model="gemma:2b")

print("Start chat with the AI model. Type 'exit' to end the chat.")

while True:
    user_input = input("You: ")
    if user_input.lower() == 'exit':
        break

    chat_history.add_message(HumanMessage(content=user_input))
    response = model.invoke(chat_history.messages)
    chat_history.add_message(AIMessage(content=response.content))


    print(f"AI: {response.content}")
    
    
print("Chat ended.")

chat_history.save()
