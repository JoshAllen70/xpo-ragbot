from dotenv import load_dotenv
import os
import openai
import pinecone

# Load in api keys
load_dotenv()
openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
pinecone_db = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))

# Connect to db
INDEX_DATABASE = "xpo-rag-chatbot-index"
index = pinecone_db.Index(INDEX_DATABASE)

###
# Functions to be used
###

# Return a vector from user input
def get_embedding(text, model="text-embedding-3-small"):
    response = openai.Embeddings.create(input=text, model=model)
    return response['data'][0]['embedding']

# Query Pinecone and return top matches of vector
def query_pinecone(query, top_k=5):
    query_vector = get_embedding(query)
    results = index.query(vector=query_vector, top_k=top_k, include_metadata=True)
    return results['matches']

# Creates a string for the matching data response
def format_context(matches):
    context = ""
    for match in matches:
        context += f"Source: {match['metadata'].get('source', 'unknown')}\n"
        context += f"Content: {match['metadata'].get('content', '')}\n\n"
    return context

# Generate a response from LLM (ChatGPT)
def generate_answer(user_query, context, model="gpt-4"):
    prompt = f"Use the following context to answer the question.\n\nContext:\n{context}\n\nQuestion: {user_query}"
    response = openai.ChatCompletion.create(
        model=model,
        messages=[{"role": "user", "content": prompt}]
    )
    return response['choices'][0]['message']['content']