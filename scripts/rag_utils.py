# rag_utils.py

from dotenv import load_dotenv
import os
from openai import OpenAI
from pinecone import Pinecone


load_dotenv()
openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
pinecone_db = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))

INDEX_NAME = "xpo-rag-chatbot-index"
index = pinecone_db.Index(INDEX_NAME)


### Helper Functions ###

# Returns the embedding vector for the given text using OpenAI embeddings.
def get_embedding(text, model="text-embedding-3-small"):
    response = openai_client.embeddings.create(input=text, model=model)
    return response.data[0].embedding
    
# Queries Pinecone for top matching vectors and returns results.
def query_pinecone(query, top_k=5):

    query_vector = get_embedding(query)
    results = index.query(
        vector=query_vector,
        top_k=top_k,
        include_metadata=True
    )

    matches = []
    for match in results['matches']:
        metadata = match.get('metadata', {})
        matches.append({
            'score': match.get('score', 0),
            'source': metadata.get('source', 'unknown'),
            'content': metadata.get('content', '')
        })

    return matches

# Formats the top matches into a string to use as context for the LLM.
def format_context(matches):
    context = ""
    for match in matches:
        context += f"Score: {match['score']:.3f}\n"
        context += f"Source: {match['source']}\n"
        context += f"Content: {match['content']}\n\n"
    return context

# Generates a response from GPT-4 given a user query and context.
def generate_answer(user_query, context, model="gpt-4"):
    prompt = f"Use the following context to answer the question.\n\nContext:\n{context}\nQuestion: {user_query}"
    response = openai_client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content