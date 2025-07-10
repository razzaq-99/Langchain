import os 
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
from dotenv import load_dotenv


load_dotenv()
os.environ["USER_AGENT"] = "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"

current_dir = os.path.dirname(os.path.abspath(__file__))
db_dir = os.path.join(current_dir, "db")
persistent_directory = os.path.join(db_dir, "chroma_db_tesla")


url = "https://www.tesla.com/"

loader = WebBaseLoader(
    web_paths=[url],
    header_template={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
)
documents = loader.load()

text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)

docs = text_splitter.split_documents(documents)

print("\n---Document Chunk Information---")
print(f"Number of Documents Chunks : {len(docs)}")
print(f"Sample Chunk:\n{docs[0].page_content}\n")


embeddings = OllamaEmbeddings(model="gemma:2b")

if not os.path.exists(persistent_directory):
    print(f"\n--- Creating vector store in {persistent_directory} ---")
    db = Chroma.from_documents(docs, embeddings, persist_directory=persistent_directory)
    print(f"--- Finished creating vector store in {persistent_directory} ---")
else:
    print(f"Vector store {persistent_directory} already exists. No need to initialize.")
    db = Chroma(persist_directory=persistent_directory, embedding_function=embeddings)


retriever = db.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 3},
)


query = "What is the latest news about Tesla?"

relevant_docs = retriever.retrieve(query)

print("\n--- Relevant Documents ---")
for i, doc in enumerate(relevant_docs, 1):
    print(f"Document {i}:\n{doc.page_content}\n")
    if doc.metadata:
        print(f"Source: {doc.metadata.get('source', 'Unknown')}\n")