import streamlit as st 
import requests
import os
from dotenv import load_dotenv

load_dotenv()

os.environ["LANGCHAIN_API_KEY"]=os.getenv("LANGCHAIN_API_KEY")
os.environ["LANGCHAIN_API_KEY"]="true"

def get_groq_response(input_text):
    response = requests.post("http://localhost:8000/essay/invoke", json = {'input':{'topic': input_text}})
    
    return response.json()['output']['content']


def get_ollama_response(input_text):
    response = requests.post("http://localhost:8000/poem/invoke", json = {'input':{'topic': input_text}})
    
    return response.json()['output']


st.title("Small Application with Grok & Ollama API")
input_text = st.text_input("write a essay on")
input_text1 = st.text_input("write a poem on")


if input_text:
    st.write(get_groq_response(input_text))
    
if input_text1:
    st.write(get_ollama_response(input_text1))