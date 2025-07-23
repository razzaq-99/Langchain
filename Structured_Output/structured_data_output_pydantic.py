from typing import TypedDict,Annotated,Optional
from langchain_ollama import ChatOllama
from dotenv import load_dotenv
import os
from pydantic import BaseModel , Field

load_dotenv()
os.environ["LANGCHAIN_TRACING_V2"] = "false"
os.environ["LANGCHAIN_API_KEY"] = ""

model = ChatOllama(model='gemma:2b')

class Reviews(BaseModel):
    key_theme :list[str]  = Field(description="Write down all the themes mentioned in the review")
    review :str = Field(description="description of the user review")
    sentiment :str = Field(description="Return user review/sentiment type")
    reviewer_name :str = Field(description="Name of the reviewer")
    pros :Optional[list[str]] = Field(default=None,description="Write down all the pros mentioned in the review")
    cons :Optional[list[str]] = Field(default=None,description="Write down all the cons mentioned in the review")
    
    
structured_model = model.with_structured_output(Reviews)

result1 = structured_model.invoke("""Overall, I wouldn't say the products are bad — there’s a lot of potential here, especially in terms of fabric and design — but some areas like quality control and sizing need improvement. I'd consider ordering again, but probably only after checking more customer reviews next time. 
                                  
    Review by Ahad Channa""")


# result2 = structured_model.invoke("I'm not sure if I can recommend this product to anyone. The quality is poor, and the sizing is off. I wouldn't buy it again.")

# result3 = structured_model.invoke("The product is very great and cost-effective,highly recommended")

# print(result1)
# print(result2)
# print(result3)

print(result1.reviewer_name)