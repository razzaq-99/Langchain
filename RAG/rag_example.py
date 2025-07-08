import os

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_core.messages import SystemMessage,HumanMessage

load_dotenv()

os.environ["LANGCHAIN_TRACING_V2"] = "false"
os.environ["LANGCHAIN_API_KEY"] = ""


current_dir = os.path.dirname(os.path.abspath(__file__))
db_dir = os.path.join(current_dir, "utils")
persistent_directory = os.path.join(db_dir, "chroma_db")


embeddings = OllamaEmbeddings(
    model="gemma:2b",
)


db = Chroma(persist_directory=persistent_directory,
            embedding_function=embeddings)

query = "How can I learn more about Langchain?"

retriever = db.as_retriever(
    search_type="similarity_score_threshold",
    search_kwargs={
        "k": 3,
        "score_threshold": 0.2
    },
)

relevant_docs = retriever.invoke(query)


combined_input = (
    "Here are some documents that might help answer the question: "
    + query
    + "\n\nRelevant Documents:\n"
    + "\n\n".join([doc.page_content for doc in relevant_docs])
    + "\n\nPlease provide an answer based only on the provided documents. If the answer is not found in the documents, respond with 'I'm not sure'."
)


model = ChatOllama(model="gemma:2b")

# Define the messages for the model
messages = [
    SystemMessage(content="You are a helpful assistant."),
    HumanMessage(content=combined_input),
]


result = model.invoke(messages)

print("\n--- Generated Response ---")
# print("Full result:")
# print(result)
print("Content only:")
print(result.content)