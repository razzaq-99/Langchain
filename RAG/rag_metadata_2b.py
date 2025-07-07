import os

from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from dotenv import load_dotenv

load_dotenv()

current_dir = os.path.dirname(os.path.abspath(__file__))
db_dir = os.path.join(current_dir, "utils")
persistent_directory = os.path.join(db_dir, "chroma_db_with_metadata")

os.environ["LANGCHAIN_TRACING_V2"] = "false"
os.environ["LANGCHAIN_API_KEY"] = ""

embeddings = OllamaEmbeddings(
    model="gemma:2a",
)

db = Chroma(persist_directory=persistent_directory,
            embedding_function=embeddings)


query = "What is The 1% Rule of atomic_habits"


retriever = db.as_retriever(
    search_type="similarity_score_threshold",
    search_kwargs={"k": 3, "score_threshold": 0.1},
)
relevant_docs = retriever.invoke(query)


print("\n--- Relevant Documents ---")
for i, doc in enumerate(relevant_docs, 1):
    print(f"Document {i}:\n{doc.page_content}\n")
    print(f"Source: {doc.metadata['source']}\n")