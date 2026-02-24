import os
from dotenv import load_dotenv
from google import genai
from pydantic import BaseModel, Field


load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
genai_client = genai.Client(api_key=GEMINI_API_KEY)


class VerificationResult(BaseModel):
    is_supported: bool = Field(
        description="True ONLY if every claim in the is fully supported by the retrieved context."
    )
    reasoning: str = Field(description="Step-by-step explanation of the evaluation.")
    hallucinated_citations: list[str] = Field(
        description="List of cited Document IDs that do not "
        "exist or do not support the claim. Empty if perfect."
    )


def verify_answer(user_query, retrieved_context, generated_answer):
    """Forces Gemini to audit the answer and return a structured Pydantic Object."""

    prompt = f"""
    You are a strict legal verification AI. Your job is to audit an answer against retrieved legal code.
    If the answer includes details not found in the context, or cites a Doc ID incorrectly, it is NOT supported.
    
    USER QUERY: {user_query}
    
    RETRIEVED CONTEXT: 
    {retrieved_context}
    
    GENERATED ANSWER TO EVALUATE:
    {generated_answer}
    """

    print("Running strict verification audit...")
    response = genai_client.models.generate_content(
        model="gemini-3.1-pro-preview",
        contents=prompt,
        config={
            "response_mime_type": "application/json",
            "response_schema": VerificationResult,
            "temperature": 0.0,  # Force maximum strictness, zero creativity
        },
    )

    # The new GenAI SDK automatically parses the JSON back into your Pydantic class!
    return response.parsed


if __name__ == "__main__":
    # Let's mock a scenario to test the Verifier without needing MongoDB right now
    test_query = "What must the landlord provide regarding temperature?"

    test_context = (
        "[Doc ID: NYC_chunk_1] The landlord must provide hot water at all times."
    )

    # We are deliberately feeding it an answer that hallucinates "heating" and a fake citation
    bad_answer = "The landlord must provide hot water (NYC_chunk_1) and central heating (NYC_chunk_99)."

    result = verify_answer(test_query, test_context, bad_answer)

    print("\n================ VERIFICATION RESULT ================\n")
    print(f"Passes Audit: {result.is_supported}")
    print(f"Reasoning:    {result.reasoning}")
    print(f"Fake Citations: {result.hallucinated_citations}")
    print("\n=====================================================\n")
