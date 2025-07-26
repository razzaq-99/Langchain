from langchain_community.chat_models import ChatOllama
from dotenv import load_dotenv
import os
from langchain_core.prompts import PromptTemplate 
# from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field


load_dotenv()

os.environ["LANGCHAIN_TRACING_V2"] = "false"
os.environ["LANGCHAIN_API_KEY"] = "" 


model = ChatOllama(model="gemma:2b")

class Person(BaseModel):
    name: str = Field(description="Name of the person")
    age: int = Field(gt=18 , description="Age of the person, must be greater than 18")
    city: str = Field(description="City where the person lives")
    

parser = PydanticOutputParser(pydantic_object=Person)

template = PromptTemplate(
    template = "Give me the name , age and city of a fictional {place} person\n {format_instruction}",
    input_variables=["place"],
    partial_variables={"format_instruction": parser.get_format_instructions()}
)

# prompt = template.invoke({"place": "Pakistani"})

# result = model.invoke(prompt)

# final_result = parser.parse(result.content)

# print(final_result)


chain = template | model | parser
result = chain.invoke({"place": "Pakistani"})
print(result)
