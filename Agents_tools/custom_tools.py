import os
from typing import Type 
# from dotenv import load_dotenv
from langchain import hub
from langchain_ollama import ChatOllama
from langchain.agents import AgentExecutor , create_tool_calling_agent
from langchain_core.tools import BaseTool 
from pydantic.v1 import BaseModel , Field


# load_dotenv()

os.environ["LANGCHAIN_TRACING_V2"] = "false"
os.environ["LANGCHAIN_API_KEY"] = ""

class SimpleSearchInput(BaseModel):
    query: str = Field(description="Should be a Search Query")
    

class MultiplyNumbersArgs(BaseModel):
    a: float = Field(description="First number to multiply")
    b: float = Field(description="Second number to multiply")
    

class SimpleSearchTool(BaseTool):
    name: str = "simple_search"
    description: str = "A simple search tool that takes a query and returns the results"
    args_schema: Type[BaseModel] = SimpleSearchInput

    def _run(self, query: str) -> str:
        """ USE THE TOOL """
        from tavily import TavilyClient
        
        api_key = os.getenv("TAVILY_API_KEY")
        Client = TavilyClient(api_key = api_key)
        results = Client.search(query)
        return f"search results for: {query}\n\n\n{results}\n"
    
    
    

class MultiplyNumbersTool(BaseTool):
    name: str = "multiply_numbers"
    description: str = "A tool that multiplies two numbers"
    args_schema: Type[BaseModel] = MultiplyNumbersArgs

    def _run(self, a: float, b: float) -> float:
        """ USE THE TOOL """
        result = a * b
        return f"The product of {a} and {b} is {result}"
    
    
    
tools = [
    SimpleSearchTool(),
    MultiplyNumbersTool(),
]

llm = ChatOllama(model="llama3")


prompt = hub.pull("hwchase17/openai-functions-agent")

agent = create_tool_calling_agent(
    llm ,
    tools,
    prompt,
)

agent_executor = AgentExecutor.from_agent_and_tools(
    agent ,
    tools,
    verbose=True,
)


response = agent_executor.invoke({"input": "Search for 'Apple Intelligence"})
print("Response : ",response)

response = agent_executor.invoke({"input": "Multiply 5 and 7"})
print("Response : ",response)
