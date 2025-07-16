from dotenv import load_dotenv
from langchain import hub
from langchain.agents import AgentExecutor, create_react_agent
from langchain_core.tools import Tool
from langchain_ollama import ChatOllama
import os
import datetime

load_dotenv()

os.environ["LANGCHAIN_TRACING_V2"] = "false"
os.environ["LANGCHAIN_API_KEY"] = ""


def get_current_time():
    """Return the current time in H:MM AM/PM format"""
    now = datetime.datetime.now()
    return now.strftime("%I:%M %p")

tools = [
    Tool(
        name="Time",
        func=get_current_time,
        description="Useful when you want to get the current time"
    )
]


prompt = hub.pull("hwchase17/react")


llm = ChatOllama(
    model="gemma:2b",
    temperature=0.1
)


agent = create_react_agent(
    llm=llm,
    tools=tools,
    prompt=prompt
)


agent_executor = AgentExecutor.from_agent_and_tools(
    agent=agent,
    tools=tools,
    verbose=True,
    handle_parsing_errors=True
)


response = agent_executor.invoke({"input": "What is the current time?"})
print("response:", response)
