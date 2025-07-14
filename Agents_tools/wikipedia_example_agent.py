from dotenv import load_dotenv
import os
from langchain.agents import AgentExecutor, create_structured_chat_agent
from langchain_ollama import ChatOllama
from langchain_core.tools import Tool
from langchain import hub
from langchain.memory import ConversationBufferMemory
from wikipedia import summary

load_dotenv()
os.environ["LANGCHAIN_TRACING_V2"] = "false"
os.environ["LANGCHAIN_API_KEY"] = "" 


def get_current_time(*args, **kwargs):
    import datetime
    return datetime.datetime.now().strftime("%I:%M %p")


def search_wikipedia(query):
    try:
        return summary(query, sentences=2)
    except Exception as e:
        return "I'm not sure."


tools = [
    Tool(
        name="Time",
        func=get_current_time,
        description="Useful for getting the current time in H:MM AM/PM format."
    ),
    Tool(
        name="Wikipedia",
        func=search_wikipedia,
        description="Useful for answering questions about general knowledge by searching Wikipedia."
    ),
]


prompt = hub.pull("hwchase17/structured-chat-agent")

llm = ChatOllama(model="gemma:2b")


memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)


agent = create_structured_chat_agent(
    llm=llm,
    tools=tools,
    prompt=prompt
)


agent_executor = AgentExecutor.from_agent_and_tools(
    agent=agent,
    tools=tools,
    memory=memory,
    verbose=True,
    handle_parsing_errors=True  
)

print("Welcome! Ask me anything. Type 'exit' to quit.\n")


while True:
    user_input = input("User: ")
    if user_input.lower() in ["exit", "quit"]:
        break

    try:
        response = agent_executor.invoke({"input": user_input})
        print("Bot:", response["output"])
    except Exception as e:
        print("Bot: Sorry, something went wrong:", str(e))
