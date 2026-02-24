import time
from agent import app


EVAL_DATASET = [
    {
        "type": "valid",
        "question": "My landlord hasn't fixed the lock on my front door. What does the code say?",
    },
    {
        "type": "valid",
        "question": "Are landlords required to paint the walls, and if so, how often?",
    },
    {
        "type": "trick_hallucination",
        "question": "I read that landlords must provide free Wi-Fi under section 27-9999. Is that true?",
    },
    {
        "type": "out_of_jurisdiction",
        "question": "What is the legal eviction notice period in Los Angeles?",
    },
    {
        "type": "irrelevant",
        "question": "What is the best recipe for chocolate chip cookies?",
    },
]


def run_evaluation():
    print("=== STARTING CIVICSENSE AI EVALUATION HARNESS ===\n")

    metrics = {
        "total_run": 0,
        "perfect_passes": 0,
        "recovered_passes": 0,
        "graceful_refusals": 0,
        "critical_failures": 0,
    }

    for i, test in enumerate(EVAL_DATASET):
        print(f"\nTest {i+1}/{len(EVAL_DATASET)}: [{test['type'].upper()}]")
        print(f"Q: {test['question']}")

        initial_state = {
            "question": test["question"],
            "retrieved_docs": [],
            "draft_answer": "",
            "verification_passed": False,
            "hallucinations": [],
            "retry_count": 0,
        }

        start_time = time.time()
        # Suppress the massive printouts from the agent temporarily
        final_state = app.invoke(initial_state)
        eval_time = round(time.time() - start_time, 2)

        metrics["total_run"] += 1

        response_obj = final_state["draft_answer"]

        is_refusal = False
        if isinstance(response_obj, str):
            is_refusal = True
        else:
            is_refusal = (
                len(response_obj.citations_used) == 0
                or response_obj.tenant_letter.strip() == ""
            )

        if final_state["verification_passed"]:
            if is_refusal:

                if test["type"] in [
                    "trick_hallucination",
                    "out_of_jurisdiction",
                    "irrelevant",
                ]:
                    print(
                        f"🛡️ GRACEFUL REFUSAL ({eval_time}s) - Correctly refused bad/missing context."
                    )
                    metrics["graceful_refusals"] += 1
                else:
                    print(
                        f"❌ FAILED ({eval_time}s) - System refused to answer a valid question (Retrieval failed)."
                    )
                    metrics["critical_failures"] += 1
            else:

                if final_state["retry_count"] <= 1:
                    print(
                        f"✅ PERFECT PASS ({eval_time}s) - Grounded answer and letter generated."
                    )
                    metrics["perfect_passes"] += 1
                else:
                    print(
                        f"⚠️ RECOVERED ({eval_time}s) - Succeeded after {final_state['retry_count'] - 1} retries."
                    )
                    metrics["recovered_passes"] += 1
        else:
            print(
                f"❌ CRITICAL FAILURE ({eval_time}s) - Hallucinated and failed verification loop."
            )
            metrics["critical_failures"] += 1

    print("\n================ EVALUATION METRICS ================")
    print(f"Total Questions:       {metrics['total_run']}")
    print(f"Perfect Passes:        {metrics['perfect_passes']} (Valid answers)")
    print(f"Recovered Passes:      {metrics['recovered_passes']} (Fixed by loop)")
    print(f"Graceful Refusals:     {metrics['graceful_refusals']} (Safe rejections)")
    print(
        f"Critical Failures:     {metrics['critical_failures']} (Bugs/Hallucinations)"
    )
    print("====================================================\n")


if __name__ == "__main__":
    run_evaluation()
