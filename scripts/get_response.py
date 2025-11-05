import os
import json
import pinecone
import rag_utils
from openai import OpenAI
from dotenv import load_dotenv

# Load in api keys
load_dotenv()
openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
pinecone_db = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))

# Connect to db
INDEX_DATABASE = "xpo-rag-chatbot-index"
index = pinecone_db.Index(INDEX_DATABASE)

user_query = input("ask a question:")
matches = query_pinecone(user_query)
context = format_context(matches)
response = generate_answer(user_query, context)

print(print("\nAnswer:\n", answer))
