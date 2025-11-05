from dotenv import load_dotenv
import os
import json
from tqdm import tqdm
from openai import OpenAI
from pinecone import Pinecone, ServerlessSpec

# Load in api keys
load_dotenv()
openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
pinecone_db = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))

# Connect to db
INDEX_DATABASE = "xpo-rag-chatbot-index"
index = pinecone_db.Index(INDEX_DATABASE)

# Open json data
with open("data/prepared_docs.json", "r", encoding="utf-8") as f:
    chunks = json.load(f)

print(f"Loaded {len(chunks)} chunks from prepared_docs.json")

def get_embedding(text: str) -> list[float]:
    response = openai_client.embeddings.create(
        input=text,
        model="text-embedding-3-small"
    )
    return response.data[0].embedding

# --- Prepare and upsert to Pinecone ---
batch_size = 50
vectors = []

for i, chunk in enumerate(tqdm(chunks, desc="Indexing")):
    # Generate vector
    embedding = get_embedding(chunk["content"])
    
    # Build vectors
    vector = {
        "id": f"{chunk['show_id']}_{chunk['audience']}_{i}",
        "values": embedding,
        "metadata": {
            "show_id": chunk["show_id"],
            "audience": chunk["audience"],
            "source": chunk["source"],
            "content": chunk["content"]
        }
    }
    vectors.append(vector)

    # Send vectors
    if len(vectors) >= batch_size:
        index.upsert(vectors=vectors)
        vectors = []

if vectors:
    index.upsert(vectors=vectors)

print("Embedded and Indexed all chunks")
