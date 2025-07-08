import os

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings

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



def query_vector_store(
    store_name, query, embedding_function, search_type, search_kwargs
):
    if os.path.exists(persistent_directory):
        print(f"\n--- Querying the Vector Store {store_name} ---")
        db = Chroma(
            persist_directory=persistent_directory,
            embedding_function=embedding_function,
        )
        retriever = db.as_retriever(
            search_type=search_type,
            search_kwargs=search_kwargs,
        )
        relevant_docs = retriever.invoke(query)
        
        print(f"\n--- Relevant Documents for {store_name} ---")
        for i, doc in enumerate(relevant_docs, 1):
            print(f"Document {i}:\n{doc.page_content}\n")
            if doc.metadata:
                print(f"Source: {doc.metadata.get('source', 'Unknown')}\n")
    else:
        print(f"Vector store {store_name} does not exist.")



query = "What is the Rule #3 of Deep Work"



print("\n--- Using Similarity Search ---")
query_vector_store("chroma_db", query,
                   embeddings, "similarity", {"k": 3})

print("\n--- Using Max Marginal Relevance (MMR) ---")
query_vector_store(
    "chroma_db",
    query,
    embeddings,
    "mmr",
    {"k": 3, "fetch_k": 20, "lambda_mult": 0.5},
)

print("\n--- Using Similarity Score Threshold ---")
query_vector_store(
    "chroma_db",
    query,
    embeddings,
    "similarity_score_threshold",
    {"k": 3, "score_threshold": 0.1},
)


print("Querying demonstrations with different search types completed.")