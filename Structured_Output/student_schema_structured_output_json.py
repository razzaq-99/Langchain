from typing import TypedDict,Annotated,Optional
from langchain_ollama import ChatOllama
from dotenv import load_dotenv
import os
from pydantic import BaseModel , Field

load_dotenv()
os.environ["LANGCHAIN_TRACING_V2"] = "false"
os.environ["LANGCHAIN_API_KEY"] = ""

model = ChatOllama(model='gemma:2b')

json_schema = {
  "title": "Review",
  "type": "object",
  "properties": {
    "key_themes": {
      "type": "array",
      "items": {
        "type": "string"
      },
      "description": "Write down all the key themes discussed in the review in a list"
    },
    "summary": {
      "type": "string",
      "description": "A brief summary of the review"
    },
    "sentiment": {
      "type": "string",
      "enum": ["pos", "neg"],
      "description": "Return sentiment of the review either negative, positive or neutral"
    },
    "pros": {
      "type": ["array", "null"],
      "items": {
        "type": "string"
      },
      "description": "Write down all the pros inside a list"
    },
    "cons": {
      "type": ["array", "null"],
      "items": {
        "type": "string"
      },
      "description": "Write down all the cons inside a list"
    },
    "name": {
      "type": ["string", "null"],
      "description": "Write the name of the reviewer"
    }
  },
  "required": ["key_themes", "summary", "sentiment"]
}

    
structured_model = model.with_structured_output(json_schema)

result1 = structured_model.invoke("""Overall, I wouldn't say the products are bad — there’s a lot of potential here, especially in terms of fabric and design — but some areas like quality control and sizing need improvement. I'd consider ordering again, but probably only after checking more customer reviews next time. 
                                  
    Review by Ahad Channa""")


# result2 = structured_model.invoke("I'm not sure if I can recommend this product to anyone. The quality is poor, and the sizing is off. I wouldn't buy it again.")

# result3 = structured_model.invoke("The product is very great and cost-effective,highly recommended")

print(result1)
# print(result2)
# print(result3)

