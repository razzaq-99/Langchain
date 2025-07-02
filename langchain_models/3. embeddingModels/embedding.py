from langchain_community.embeddings import OllamaEmbeddings
from dotenv import load_dotenv
load_dotenv()


embeddings = OllamaEmbeddings(model="gemma:2b")

document = ["Hey there, I am Abdul, a software engineer. I love to code and build amazing applications.",
           "I am passionate about technology and always eager to learn new things.",
           "In my free time, I enjoy reading books and exploring new programming languages.",
           "I believe in continuous learning and strive to improve my skills every day.",]

# result = embeddings.embed_query("My name is Abdul, I am a software engineer and I love to code.")

result = embeddings.embed_documents(document)

print(result)
