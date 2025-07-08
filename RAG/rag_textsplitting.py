import os

from langchain_text_splitters import (
    RecursiveCharacterTextSplitter,
    CharacterTextSplitter,
    TokenTextSplitter,
    SentenceTransformersTokenTextSplitter,
    TextSplitter
)
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma

load_dotenv()

current_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(current_dir, "utils", "Deep_Work.txt")
db_dir = os.path.join(current_dir, "utils")

if not os.path.exists(file_path):
    raise FileNotFoundError(f"File not found: {file_path}")

loader = TextLoader(file_path, encoding="utf-8")
documents = loader.load()

embeddings = OllamaEmbeddings(model="gemma:2a")

def create_vector_store(docs, store_name):
    persistent_directory = os.path.join(db_dir, store_name)
    if not os.path.exists(persistent_directory):
        print(f"---Creating Vector Store {store_name}---")
        db = Chroma.from_documents(
            documents=docs,
            embedding_function=embeddings,
            persist_directory=persistent_directory
        )
        print(f"---Vector Store {store_name} created successfully---")
    else:
        print(f"---Vector Store {store_name} already exists---")

# Character-based splitting
print("\n---Character Text Splitter---")
character_splitter = CharacterTextSplitter(
    separator="\n",
    chunk_size=1000,
    chunk_overlap=200
)
character_docs = character_splitter.split_documents(documents)
create_vector_store(character_docs, "chroma_db_character")

# Sentence-based splitting
print("\n---Sentence Text Splitter---")
sentence_splitter = SentenceTransformersTokenTextSplitter(
    separator="\n",
    chunk_size=1000,
    chunk_overlap=200
)
sentence_docs = sentence_splitter.split_documents(documents)
create_vector_store(sentence_docs, "chroma_db_sentence")

# Token-based splitting
print("\n---Token Text Splitter---")
token_splitter = TokenTextSplitter(
    separator="\n",
    chunk_size=1000,
    chunk_overlap=200
)
token_docs = token_splitter.split_documents(documents)
create_vector_store(token_docs, "chroma_db_token")

# Recursive character-based splitting
print("\n---Recursive Character Text Splitter---")
recursive_splitter = RecursiveCharacterTextSplitter(
    separators=["\n\n", "\n", " ", ""],
    chunk_size=1000,
    chunk_overlap=200
)
recursive_docs = recursive_splitter.split_documents(documents)
create_vector_store(recursive_docs, "chroma_db_recursive")

# Custom splitter
print("\n---Custom Text Splitting---")

class CustomTextSplitter(TextSplitter):
    def split_text(self, text):
        return text.split("\n\n")

customsplitter = CustomTextSplitter()
custom_docs = customsplitter.split_documents(documents)
create_vector_store(custom_docs, "chroma_db_custom")

# Query vector store
def query_vector_store(store_name, query):
    persistent_directory = os.path.join(db_dir, store_name)
    if os.path.exists(persistent_directory):
        print(f"\n--- Querying the vector store {store_name}")
        db = Chroma(
            persist_directory=persistent_directory,
            embedding_function=embeddings
        )

        retriever = db.as_retriever(
            search_type="similarity_score_threshold",
            search_kwargs={"k": 3, "score_threshold": 0.1},
        )

        relevant_docs = retriever.invoke(query)

        print(f"\n--- Relevant Documents for {store_name}")
        for i, doc in enumerate(relevant_docs, 1):
            print(f"Document {i}:\n{doc.page_content}\n")
            if doc.metadata:
                print(f"Source: {doc.metadata.get('source', 'Unknown')}\n")
    else:
        print(f"Vector Store {store_name} does not exist.")

# Query execution
query = "What is the Rule #3 of Deep Work"

query_vector_store("chroma_db_sentence", query)
query_vector_store("chroma_db_recursive", query)
query_vector_store("chroma_db_custom", query)
query_vector_store("chroma_db_token", query)
query_vector_store("chroma_db_character", query)


