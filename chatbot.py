import streamlit as st
from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from dotenv import load_dotenv
import os


load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")


llm = ChatGroq(
    groq_api_key=os.getenv("GROQ_API_KEY"),
    model_name="llama3-70b-8192"  # ✅ or llama3-8b-8192, or gemma-7b-it
)



st.set_page_config(page_title="Abdul Chatbot")
st.title("💬 Abdul Chatbot")
st.markdown("Ask anything!")


prompt = PromptTemplate.from_template("Q: {question} \nA:")


chain = LLMChain(llm=llm, prompt=prompt)


user_input = st.text_input("Ask a question:")

if user_input:
    response = chain.invoke({"question": user_input})
    st.write("Answer: ", response["text"])
