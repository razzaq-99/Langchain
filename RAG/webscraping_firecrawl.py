import os
from dotenv import load_dotenv
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import FirecrawlLoader
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings

load_dotenv()

current_dir = os.path.dirname(os.path.abspath(__file__))
db_dir = os.path.join(current_dir, "db")
persistent_directory = os.path.join(db_dir, "chroma_db_firecrawl")

def create_vector_store():
    """Crawl the website, split the content, create embeddings, and persist the vector store."""
    
    api_key = os.getenv("FIRECRAWL_API_KEY")
    if not api_key:
        raise ValueError("FIRECRAWL_API_KEY environment variable not set")

    print("Begin crawling the website...")
    loader = FirecrawlLoader(
        api_key=api_key,
        url="https://tesla.com",
        crawl_links=True,  # Set to False if you only want the main page
        max_depth=1        # Optional: how deep to crawl
    )
    docs = loader.load()
    print("Finished crawling the website.")

    # Clean up metadata
    for doc in docs:
        for key, value in doc.metadata.items():
            if isinstance(value, list):
                doc.metadata[key] = ", ".join(map(str, value))

    # Split text into chunks
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    split_docs = text_splitter.split_documents(docs)

    print("\n--- Document Chunks Information ---")
    print(f"Number of document chunks: {len(split_docs)}")
    print(f"Sample chunk:\n{split_docs[0].page_content}\n")

    # Generate embeddings
    embeddings = OllamaEmbeddings(model="gemma:2b")

    # Create and persist vector store
    print(f"\n--- Creating vector store in {persistent_directory} ---")
    db = Chroma.from_documents(
        split_docs,
        embeddings,
        persist_directory=persistent_directory
    )
    print(f"--- Finished creating vector store in {persistent_directory} ---")

# Initialize vector store if not already created
if not os.path.exists(persistent_directory):
    create_vector_store()
else:
    print(f"Vector store {persistent_directory} already exists. No need to initialize.")

# Load the vector store for querying
embeddings = OllamaEmbeddings(model="gemma:2b")
db = Chroma(persist_directory=persistent_directory, embedding_function=embeddings)

def query_vector_store(query):
    """Query the vector store with the specified question."""
    retriever = db.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 3},
    )
    relevant_docs = retriever.invoke(query)

    print("\n--- Relevant Documents ---")
    for i, doc in enumerate(relevant_docs, 1):
        print(f"Document {i}:\n{doc.page_content}\n")
        if doc.metadata:
            print(f"Source: {doc.metadata.get('source', 'Unknown')}\n")

# Example query
query = "tesla intelligence"
query_vector_store(query)
