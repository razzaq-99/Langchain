import os

from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings


current_dir = os.path.dirname(os.path.abspath(__file__))
persistent_directory = os.path.join(current_dir, "utils", "chroma_db")

os.environ["LANGCHAIN_TRACING_V2"] = "false"
os.environ["LANGCHAIN_API_KEY"] = "" 

embeddings = OllamaEmbeddings(
    model="gemma:2b",
)


db = Chroma(persist_directory=persistent_directory,
            embedding_function=embeddings)


query = "What is Rule #3 of Deep Work?"


retriever = db.as_retriever(
    search_type="similarity_score_threshold",
    search_kwargs={"k": 1, "score_threshold": 0.4},
)
relevant_docs = retriever.invoke(query)


print("\n--- Relevant Documents ---")
for i, doc in enumerate(relevant_docs, 1):
    print(f"Document {i}:\n{doc.page_content}\n")
    if doc.metadata:
        print(f"Source: {doc.metadata.get('source', 'Unknown')}\n")