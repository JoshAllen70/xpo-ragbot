from dotenv import load_dotenv
import os
from openai import OpenAI
from pinecone import Pinecone

# Load in api keys
load_dotenv()
openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
pinecone_db = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))

# Connect to db
INDEX_DATABASE = "xpo-rag-chatbot-index"
index = pinecone_db.Index(INDEX_DATABASE)

# Function to grab embeddings
def get_embedding(text):
    response = openai_client.embeddings.create(
        input=text,
        model="text-embedding-3-small"
    )
    return response.data[0].embedding

# Testing Query (can change but use verifiable data)
exhibitor_query = "What are the pack in and pack out hours?"
query_embedding = get_embedding(exhibitor_query)

# Search Pinecone
results = index.query(
    vector=query_embedding,
    top_k=5,
    include_metadata=True
)

# Display matches
print("\n🔍 Top Matches:\n")
for match in results['matches']:
    print(f"Score: {match['score']:.3f}")
    print(f"Show: {match['metadata']['show_id']} | Audience: {match['metadata']['audience']}")
    print(f"Source: {match['metadata']['source']}")
    print(f"Content:\n{match['metadata']['content'][:300]}...")
    print("-" * 80)

