from langchain_ollama import ChatOllama
from dotenv import load_dotenv
import os
from langchain_core.prompts import PromptTemplate
from langchain.output_parsers import StructuredOutputParser , ResponseSchema

load_dotenv()

os.environ["LANGCHAIN_TRACING_V2"] = "false"
os.environ["LANGCHAIN_API_KEY"] = "" 


model = ChatOllama(model="gemma:2b")

schema = [
    ResponseSchema(name="fact_1", description="Fact 1 about the topic"),
    ResponseSchema(name="fact_2", description="Fact 2 about the topic"),
    ResponseSchema(name="fact_3", description="Fact 3 about the topic"),
    ResponseSchema(name="fact_4", description="Fact 4 about the topic"),
    ResponseSchema(name="fact_5", description="Fact 5 about the topic"),
]

parser = StructuredOutputParser.from_response_schemas(schema)

temp1 = PromptTemplate(
    template = "Give 5 facts about the {topic}\n{format_instruction}",
    input_variables=['topic'],
    partial_variables={"format_instruction":parser.get_format_instructions()}
)


# prompt = temp1.invoke({'topic':'Influence of AI'})
# result = model.invoke(prompt)
# final_result = parser.parse(result.content)



chain = temp1 | model | parser
final_result = chain.invoke({'topic':'Influence of AI'})
print(final_result)