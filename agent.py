from typing import TypedDict, List
from langgraph.graph import StateGraph, END

from basic_rag import embed_query, retrieve_legal_context, generate_grounded_answer
from verifier import verify_answer


class AgenticRAGState(TypedDict):
    """The memory object passed between every node in the LangGraph."""

    question: str
    retrieved_docs: List[dict]  # Will hold the chunks from MongoDB
    draft_answer: str  # The current attempt from Gemini
    verification_passed: bool  # True/False from your Verifier
    hallucinations: List[str]  # The fake citations to avoid next time
    retry_count: int  # To prevent infinite loops


def retrieve_node(state: AgenticRAGState):
    print(f"\n[NODE: Retriever] Attempt {state.get('retry_count', 0) + 1}...")
    question = state["question"]

    vector = embed_query(question)
    docs = retrieve_legal_context(vector)

    return {"retrieved_docs": docs}


def generate_node(state: AgenticRAGState):
    print("[NODE: Generator] Drafting answer based strictly on retrieved code...")
    docs = state["retrieved_docs"]
    question = state["question"]

    answer = generate_grounded_answer(question, docs)

    return {"draft_answer": answer}


def verify_node(state: AgenticRAGState):
    print("[NODE: Verifier] Auditing the draft for hallucinations...")
    question = state["question"]
    docs = state["retrieved_docs"]
    answer = state["draft_answer"]

    context_string = "".join(
        [f"[Doc ID: {d['document_id']}] {d['content']}\n" for d in docs]
    )

    result = verify_answer(question, context_string, answer)

    new_retry_count = state.get("retry_count", 0) + 1
    verification_passed = (
        result.get("is_supported", False) if isinstance(result, dict) else False
    )
    hallucinations = (
        result.get("hallucinated_citations", []) if isinstance(result, dict) else []
    )
    if result:
        return {
            "verification_passed": verification_passed,
            "hallucinations": hallucinations,
            "retry_count": new_retry_count,
        }


def should_continue(state: AgenticRAGState):
    if state["verification_passed"]:
        print(">>> VERDICT: PASS. Answer is perfectly grounded. Exiting loop.")
        return "end"
    elif state["retry_count"] >= 2:
        print(">>> VERDICT: FAIL. Max retries reached. Forcing graceful refusal.")
        return "max_retries"
    else:
        print(
            f">>> VERDICT: FAIL. Hallucinations detected: {state['hallucinations']}. Forcing retry..."
        )
        return "retry"


workflow = StateGraph(AgenticRAGState)

workflow.add_node("retriever", retrieve_node)
workflow.add_node("generator", generate_node)
workflow.add_node("verifier", verify_node)


workflow.set_entry_point("retriever")
workflow.add_edge("retriever", "generator")
workflow.add_edge("generator", "verifier")

workflow.add_conditional_edges(
    "verifier", should_continue, {"end": END, "retry": "retriever", "max_retries": END}
)

app = workflow.compile()


if __name__ == "__main__":
    print("\n=== STARTING CIVICSENSE AI ORCHESTRATOR ===")

    test_question = "My ceiling is leaking and the landlord is ignoring me."

    initial_state: AgenticRAGState = {
        "question": test_question,
        "retrieved_docs": [],
        "draft_answer": "",
        "verification_passed": False,
        "hallucinations": [],
        "retry_count": 0,
    }

    final_state = app.invoke(initial_state)

    print("\n================ FINAL SYSTEM OUTPUT ================\n")
    if final_state["verification_passed"]:
        response_obj = final_state["draft_answer"]

        print(" LEGAL ACTION PLAN:")
        print(response_obj.legal_action_plan)
        print("\n-----------------------------------------------------")
        print(" DRAFTED TENANT LETTER:")
        print(response_obj.tenant_letter)
        print("\n-----------------------------------------------------")
        print(f" CITATIONS USED: {response_obj.citations_used}")
    else:
        print(
            "SYSTEM REFUSAL: I cannot confidently answer this question based on the retrieved NYC code."
        )
    print("\n=====================================================\n")
