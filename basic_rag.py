import os
from dotenv import load_dotenv
from google import genai
from pymongo import MongoClient
from pymongo.server_api import ServerApi
from pydantic import BaseModel, Field

load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
MONGODB_URI = os.getenv("MONGODB_URI")


genai_client = genai.Client(api_key=GEMINI_API_KEY)
mongo_client = MongoClient(MONGODB_URI, server_api=ServerApi("1"))
db = mongo_client["civicsense_db"]
collection = db["legal_corpus"]


class LegalResponse(BaseModel):
    legal_action_plan: str = Field(
        description="Step-by-step advice for the tenant based ONLY on the retrieved context."
    )
    tenant_letter: str = Field(
        description="A formal, polite, but firm notice to the landlord citing the specific legal codes retrieved."
    )
    citations_used: list[str] = Field(
        description="List of Doc IDs explicitly used in the plan and letter."
    )


def embed_query(query_text):
    """Converts the user's question into a vector of 768 dimensions"""
    try:
        result = genai_client.models.embed_content(
            model="gemini-embedding-001",
            contents=query_text,
            config={"output_dimensionality": 768},
        )
        if result and result.embeddings:
            return result.embeddings[0].values
        return None
    except Exception as e:
        print(f"Error embedding query: {e}")
        return None


def retrieve_legal_context(query_vector):
    """Searches MongoDb for top 3 most relevant legal chunks"""

    pipeline = [
        {
            "$vectorSearch": {
                "index": "vector_index",
                "path": "embeddings",
                "queryVector": query_vector,
                "numCandidates": 50,
                "limit": 3,
            }
        },
        {
            "$project": {
                "_id": 0,
                "content": 1,
                "document_id": 1,
                "score": {"$meta": "vectorSearchScore"},
            }
        },
    ]

    results = list(collection.aggregate(pipeline))
    return results


def generate_grounded_answer(user_query, retrieved_docs):
    """Forces Gemini to output a structured action plan and drafted letter."""

    context_string = ""
    for doc in retrieved_docs:
        context_string += f"[Doc ID: {doc['document_id']}]\n{doc['content']}\n\n"

    prompt = f"""
    You are an expert legal AI advocate for tenants. 
    Based ONLY on the provided legal context, generate a Legal Action Plan and draft a formal Tenant Notice Letter to the landlord.
    
    Rules:
    1. Do not hallucinate outside laws. 
    2. You MUST cite the exact [Doc ID] in the body of the letter so the landlord sees the proof.
    3. If the context does not contain enough info to draft a valid letter, state that in the action plan and leave the letter blank.
    
    LEGAL CONTEXT:
    {context_string}
    
    USER QUESTION: 
    {user_query}
    """

    print("\nDrafting Action Plan and Legal Letter...")

    response = genai_client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt,
        config={
            "response_mime_type": "application/json",
            "response_schema": LegalResponse,
            "temperature": 0.1,
        },
    )

    return response.parsed


if __name__ == "__main__":
    query = "What is the general purpose or intent of this housing code?"

    print(f"Query: {query}")
    vector = embed_query(query)

    if vector:
        docs = retrieve_legal_context(vector)
        print(f"Retrieved {len(docs)} documents.")

        answer = generate_grounded_answer(query, docs)
        print("\n--- FINAL ANSWER ---")
        print(answer)
    else:
        print("Failed to generate embedding for the query.")
