from langchain_ollama import ChatOllama
from dotenv import load_dotenv
import os
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser

load_dotenv()

os.environ["LANGCHAIN_TRACING_V2"] = "false"
os.environ["LANGCHAIN_API_KEY"] = "" 


model = ChatOllama(model="gemma:2b")

parser = JsonOutputParser()

temp1 = PromptTemplate(
    template = "Give me the name , age and city of fictional person\n{format_instruction}",
    input_variables=[],
    partial_variables={"format_instruction":parser.get_format_instructions()}
)

# prompt = temp1.format()
# result = model.invoke(prompt)
# result_json = parser.parse(result.content)

chain = temp1 | model | parser
result_json = chain.invoke({})

print(result_json)
print(result_json["name"])
print(result_json["age"])
print(result_json["city"])
print(type(result_json))