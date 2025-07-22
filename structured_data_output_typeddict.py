from typing import TypedDict
from langchain_ollama import ChatOllama
from dotenv import load_dotenv
import os

load_dotenv()
os.environ["LANGCHAIN_TRACING_V2"] = "false"
os.environ["LANGCHAIN_API_KEY"] = ""

model = ChatOllama(model='gemma:2b')

class Reviews(TypedDict):
    review: str
    sentiment: str
    
structured_model = model.with_structured_output(Reviews)

result1 = structured_model.invoke("Overall, I wouldn't say the products are bad — there’s a lot of potential here, especially in terms of fabric and design — but some areas like quality control and sizing need improvement. I'd consider ordering again, but probably only after checking more customer reviews next time.")

result2 = structured_model.invoke("I'm not sure if I can recommend this product to anyone. The quality is poor, and the sizing is off. I wouldn't buy it again.")

result3 = structured_model.invoke("The product is very great and cost-effective,highly recommended")

print(result1)
print(result2)
print(result3)

print(result1["sentiment"])
print(result2["sentiment"])
print(result3["sentiment"])