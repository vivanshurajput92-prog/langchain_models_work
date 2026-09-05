from langchain_huggingface import HuggingFaceEndpointEmbeddings
from dotenv import load_dotenv

load_dotenv()

embeddings = HuggingFaceEndpointEmbeddings(
    model="sentence-transformers/all-MiniLM-L6-v2",
    task="feature-extraction"   
)

query_embedding = embeddings.embed_query("My name is Vivanshu chauhan.")

documents = [
    "New Delhi is the capital of India.",
    "Mumbai is the financial center of India."
]

document_embedding = embeddings.embed_documents(documents)

print(document_embedding)