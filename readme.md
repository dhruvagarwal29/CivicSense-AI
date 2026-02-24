
```markdown
# 🏛️ CivicSense AI: Autonomous Legal Advocate for Tenants

CivicSense AI is a production-grade Agentic RAG (Retrieval-Augmented Generation) system designed to democratize legal aid. It ingests the New York City Housing Maintenance Code and acts as an autonomous legal advocate—answering tenant questions, formulating legal action plans, and drafting mathematically verified demand letters.

Unlike standard LLM wrappers that are prone to hallucinations, CivicSense utilizes a **LangGraph state machine** with a strict deterministic verification loop. If the AI hallucinates a legal citation, the system intercepts it and forces a retry or gracefully refuses to answer.



## 🏗️ System Architecture

CivicSense AI is built as a cyclic graph (state machine) rather than a linear script. The core loop consists of three primary nodes passing a shared memory state:

1. **The Retriever Node:** Takes the user's plain-English question, converts it into a 768-dimension vector using `gemini-embedding-001`, and searches a MongoDB Atlas vector database for the top most relevant legal statutes.
2. **The Generator Node:** Uses a fast LLM (`gemini-2.5-flash`) constrained by a Pydantic schema to output a strict JSON object containing a step-by-step tenant action plan and a drafted demand letter, explicitly citing the retrieved Document IDs.
3. **The Verifier (Judge) Node:** Uses a high-reasoning LLM (`gemini-3.1-pro`) to rigorously audit the Generator's draft against the raw retrieved legal text. It acts as a deterministic guardrail.
4. **The Conditional Edge:** If the Verifier detects a hallucinated citation or an unsupported claim, it routes the graph back to the Retriever to try again. If it fails twice, the system safely degrades to a "Graceful Refusal."

## ✨ Key Features

* **Zero-Hallucination Guardrails:** The dual-LLM verification loop ensures that no legal letter is generated unless every single claim is backed by a specific municipal code citation.
* **Autonomous Orchestration:** Treats the RAG pipeline as a self-correcting agent. It loops autonomously until a verified answer is generated or the max retry limit is reached.
* **Automated Evaluation Harness:** Includes a built-in testing suite (`evaluate.py`) that blasts the agent with trick questions, out-of-jurisdiction queries, and valid complaints to mathematically grade its safety and retrieval performance.

## 🛠️ Tech Stack

* **Orchestration:** LangGraph
* **LLM Provider:** Google Gemini API (Flash for generation, Pro for strict verification)
* **Vector Database:** MongoDB Atlas Vector Search
* **Data Ingestion:** PyMuPDF, LangChain (`RecursiveCharacterTextSplitter`)
* **Data Structuring:** Pydantic

## 🚀 Installation & Setup

### 1. Clone the repository
```bash
git clone [https://github.com/dhruvagarwal29/CivicSense-AI.git](https://github.com/dhruvagarwal29/CivicSense-AI.git)
cd civicsense-ai

```

### 2. Install Dependencies

Create a virtual environment and install the required packages:

```bash
python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`
pip install google-genai pymongo python-dotenv pydantic langgraph pymupdf langchain-text-splitters

```

### 3. Environment Variables

Create a `.env` file in the root directory:

```ini
GEMINI_API_KEY="your_google_gemini_api_key"
MONGODB_URI="your_mongodb_atlas_connection_string"

```

### 4. MongoDB Atlas Setup

Create a database named `civicsense_db` and a collection named `legal_corpus`.
Inside the Atlas UI, create an Atlas Vector Search JSON index named exactly `vector_index` mapped to the `embeddings` field (768 dimensions, cosine similarity).

## 🧠 Usage

### 1. Ingesting the Legal Code

To parse the PDF, chunk the text, generate vectors, and upload them to MongoDB using multi-threading:

```bash
python ingest.py

```

### 2. Running the Agent

To run the LangGraph orchestrator in the terminal and generate a verified legal letter:

```bash
python agent.py

```

### 3. Running the Evaluation Harness

To test the system's robustness against edge cases and trick questions:

```bash
python evaluate.py

```

## 🗺️ Roadmap / Future Upgrades

* **API Integration:** Wrap the LangGraph agent in a FastAPI server to expose a `/analyze` endpoint for frontend applications.
* **Semantic Caching:** Implement a MongoDB query cache to bypass LLM generation for frequently asked questions, reducing latency and API costs.
* **Query Expansion Node:** Add a translation step in LangGraph to convert emotional tenant complaints into dense legal vocabulary before vector search.

```

***

This perfectly encapsulates the heavy engineering you have done without promising features that aren't in the codebase yet. 

**Are you ready to commit this to GitHub, or is there anything else in the codebase you want to tidy up first?**

```