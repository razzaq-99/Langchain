from fastapi import FastAPI
from langchain_groq import ChatGroq
from langchain.prompts import ChatPromptTemplate
from langserve import add_routes
from dotenv import load_dotenv
import os
import uvicorn
from langchain_ollama import OllamaLLM  

os.environ["LANGCHAIN_API_KEY"]=os.getenv("LANGCHAIN_API_KEY")
os.environ["LANGCHAIN_API_KEY"]="true"

load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")


app = FastAPI(
    title="Abdul Chatbot API",
    description="A simple chatbot API using Groq and LangChain.",
    version="1.0.0"
)


groq_model = ChatGroq(
    groq_api_key=groq_api_key,
    model="mistral"  
)


ollama_model = OllamaLLM(model="gemma:2b")



prompt1 = ChatPromptTemplate.from_template("Write me an essay about {topic} with 100 words.")
prompt2 = ChatPromptTemplate.from_template("Write a poem on {topic} for a beautiful girl with 100 words.")


add_routes(app, groq_model, path="/chatgroq")
add_routes(app, prompt1 | groq_model, path="/essay")
add_routes(app, prompt2 | ollama_model, path="/poem")


uvicorn.run(app, host="localhost", port=8000)

