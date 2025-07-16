from langchain import hub
from langchain.agents import AgentExecutor,create_tool_calling_agent
from langchain_core.tools import StructuredTool , Tool
from langchain_ollama import ChatOllama
from pydantic import BaseModel, Field
import os 


os.environ["LANGCHAIN_TRACING_V2"] = "false"
os.environ["LANGCHAIN_API_KEY"] = ""

def greet_user(name: str) -> str:
    """Greet the user with their name."""
    return f"Hello, {name}!"


def reverse_string(string: str) -> str:
    """Reverse the given string."""
    return string[::-1]


def concatenate_strings(string1: str, string2: str) -> str:
    """Concatenate two strings."""
    return string1 + string2


class ConcatenateStrings(BaseModel):
    string1: str = Field(description="The first string to concatenate.")
    string2: str = Field(description="The second string to concatenate.")
    


tools = [
    Tool(
        name="greet_user",
        description="Greet the user with their name.",
        func=greet_user,
    ),
    Tool(
        name="reverse_string",
        description="Reverse the given string.",
        func=reverse_string,
    ),
    StructuredTool.from_function(
        name="concatenate_strings",
        description="Concatenate two strings.",
        args_schema=ConcatenateStrings,
        func=concatenate_strings,
    ),
]


llm = ChatOllama(model="gemma:2b")

prompt = hub.pull("hwchase17/react-chat")

agent = create_tool_calling_agent(
    llm = llm ,
    tools=tools,
    prompt= prompt
)


agent_executor = AgentExecutor.from_agent_and_tools(
    agent=agent,
    tools=tools,
    verbose=True,
)

response = agent_executor.invoke({"input":"Greet Alice"})
print("Response : ",response)

response = agent_executor.invoke({"input":"Reverse the string 'Hello, World!'"})
print("Response : ",response)

response = agent_executor.invoke({"input":"Concatenate the strings 'Hello' and 'World'"})
print("Response : ",response)