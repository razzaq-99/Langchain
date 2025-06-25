import streamlit as st
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_community.llms import ollama
import os

os.environ["LANGCHAIN_API_KEY"]=os.getenv("LANGCHAIN_API_KEY")
os.environ["LANGCHAIN_API_KEY"]="true"

llm = ollama.Ollama(
    model="llama2"  )


st.set_page_config(page_title="Abdul Chatbot")
st.title("💬 Abdul Chatbot")
# st.markdown("Ask anything!")


prompt = PromptTemplate.from_template("Q: {question} \nA:")


chain = LLMChain(llm=llm, prompt=prompt)


user_input = st.text_input("Ask a question:")


if user_input:
    response = chain.run(question=user_input)
    st.text(response)
    st.write("🤖 Abdul")
    st.write("Answer:", response)

st.write("Powered by Ollama and LangChain")
st.write("Made with ❤️ by Abdul Razzaq")
