import os
import time
from dotenv import load_dotenv
from google import genai
from pymongo import MongoClient
from pymongo.server_api import ServerApi
from parse_pdf import chunk_nyc_code
from concurrent.futures import ThreadPoolExecutor, as_completed


load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
MONGODB_URI = os.getenv("MONGODB_URI")

genai_client = genai.Client(api_key=GEMINI_API_KEY)
mongo_client = MongoClient(MONGODB_URI, server_api=ServerApi("1"))
db = mongo_client["civicsense_db"]
collection = db["legal_corpus"]


def generate_embeddings(text):
    """Calls Gemini to convert text into a 768-dimension vector"""
    try:
        result = genai_client.models.embed_content(
            model="gemini-embedding-001",
            contents=text,
            config={"output_dimensionality": 768},
        )
        if result and result.embeddings:
            return result.embeddings[0].values
    except Exception as e:
        print(f"\nAPI Error (Might be rate limit): {e}")
        return None


def process_single_chunk(chunk):
    """The function each thread will run independently."""
    vector = generate_embeddings(chunk["content"])
    if vector:
        chunk["embeddings"] = vector
        # PyMongo is thread-safe, so multiple threads can write at once
        collection.update_one(
            {"document_id": chunk["document_id"]}, {"$set": chunk}, upsert=True
        )
        return True
    return False


def ingest_data_multithreaded(pdf_path):
    print("Step 1: Parsing PDF...")
    chunks = chunk_nyc_code(pdf_path)[21:]

    if not chunks:
        print("No chunks found. Exiting.")
        return

    print(f"Step 2: Multithreading embeddings for {len(chunks)} chunks...")

    success_count = 0

    # We use 5 workers. This means 5 simultaneous API calls at any given time.
    MAX_WORKERS = 5

    start_time = time.time()

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_to_chunk = {
            executor.submit(process_single_chunk, chunk): chunk for chunk in chunks
        }

        for future in as_completed(future_to_chunk):
            if future.result():
                success_count += 1
                print(
                    f"\rProgress: {success_count} / {len(chunks)} uploaded...", end=""
                )

    end_time = time.time()
    print(
        f"\n\n SUCCESS: Uploaded {success_count} chunks in {round(end_time - start_time, 2)} seconds!"
    )


if __name__ == "__main__":
    try:
        mongo_client.admin.command("ping")
        print("Pinged your deployment. Successfully connected to MongoDB!\n")
    except Exception as e:
        print(
            f"Database connection failed. Check your MONGODB_URI in .env.\nError: {e}"
        )
        exit()

    ingest_data_multithreaded("data/nyc_hmc.pdf")
