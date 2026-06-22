import json
import os
import time

from dotenv import load_dotenv
from google import genai
from google.genai import types


def main():
    load_dotenv()
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print("Error: Please set the GEMINI_API_KEY environment variable.")
        return

    client = genai.Client(api_key=api_key)

    base_prompt = """You are an expert A-Level Mathematics examiner, pedagogical specialist, and Python programmer. Your task is to generate mathematically flawless, highly original exam questions and detailed mark schemes.

You have access to a Python Code Execution environment. You MUST use it to act as a deterministic oracle and verify the mathematical soundness of your question BEFORE outputting the final response.

### THE WORK-BACKWARDS PROTOCOL (MANDATORY)
1. **Answer First:** Do not write the question text first. Choose the final, clean, "nice" answers first (e.g., integers, simple fractions, clear surds).
2. **Reverse Engineer:** Write Python code (using `sympy`, `numpy`, or `math`) to calculate the required constants, coefficients, or givens that lead to your chosen answers.
3. **Execute & Verify:** Run the code. Check for unintended edge cases:
   - Are there multiple roots/intersections when only one is expected?
   - Does a calculus derivative simplify so much that the question loses its intended difficulty?
   - Are physical/statistical constraints violated (e.g., negative friction, empty statistical critical regions)?
4. **Finalize:** Only after the Python execution confirms the math is perfect and the values are sound, construct the final question text and mark scheme.

### DOMAIN GUARDRAILS
- **Pure Math:** Verify multi-step integrals, geometric series sums, and ensure derivatives don't collapse to trivial linear results.
- **Mechanics:** Ensure Normal Reaction R >= 0, static friction <= mu*R.
- **Statistics:** Verify all critical values and probabilities using Python. Never hand-sum expected values.

### OUTPUT SCHEMA & CRITICAL RULES
Respond ONLY with a valid JSON object. No prose, no markdown fences outside the JSON, no working notes.

- **JSON/LaTeX Escaping [CRITICAL]:** Every LaTeX backslash must be written as a DOUBLE backslash inside JSON strings.
  - Correct: "Find $\\frac{dy}{dx}$"
  - Wrong: "Find $\\frac{dy}{dx}$"
- **Format:**
{
  "topic": "String",
  "difficulty": "foundation" | "higher" | "extension",
  "question_latex": "The exam question formatted in LaTeX using $ for inline and $$ for display.",
  "markscheme_latex": "The step-by-step solution formatted in LaTeX. Use double backslashes for all commands like \\\\int or \\\\frac.",
  "verification_summary": "A brief 1-sentence explanation of what you reverse-engineered and verified with Python."
}
"""

    method_1_prompt = base_prompt + "\nGenerate EXACTLY 1 question."

    # We modify the output schema slightly for method 2 to expect a list of 3 questions
    method_2_prompt = (
        base_prompt.replace("- **Format:**\n{", "- **Format:**\n[\n  {").replace(
            "}\n", "  }\n]\n(Return a list of EXACTLY 3 such objects)"
        )
        + "\nGenerate EXACTLY 3 questions."
    )

    # We will use gemini-2.5-flash as the "gemini-lite" model, as it is the standard fast/lite model.
    # We enable code execution to satisfy the prompt's requirements.
    generation_model = "gemini-3.1-flash-lite"
    evaluation_model = "gemini-3.5-flash"

    config = types.GenerateContentConfig(
        temperature=0.7,
        tools=[{"code_execution": {}}],
    )

    def get_response_text(response):
        text = ""
        try:
            if response.candidates and response.candidates[0].content and response.candidates[0].content.parts:
                for part in response.candidates[0].content.parts:
                    if part.text:
                        text += part.text
            else:
                text = response.text
        except Exception:
            text = str(response)
        return text

    def extract_json(text):
        if not text:
            raise ValueError("Empty text provided for JSON extraction")
        if "```json" in text:
            text = text.split("```json")[1].split("```")[0]
        elif "```" in text:
            text = text.split("```")[1].split("```")[0]
        return json.loads(text.strip())

    import traceback

    print("=== Starting Method 1: 1 question at a time (45 iterations) ===")
    method_1_results = []
    for i in range(45):
        print(f"  Method 1 - Iteration {i + 1}/45...")
        try:
            response = client.models.generate_content(
                model=generation_model,
                contents=method_1_prompt,
                config=config,
            )
            # Context is "cleared" inherently because we are calling generate_content (stateless)
            try:
                response_text = get_response_text(response)
                data = extract_json(response_text)
                method_1_results.append(data)
            except json.JSONDecodeError as e:
                print(f"    Failed to parse JSON on iteration {i+1}: {e}")
                print(f"    Raw output: {response_text}")
        except Exception as e:
            print(f"    Error on iteration {i+1}: {e}")
            traceback.print_exc()
        time.sleep(2)  # avoid rate limits

    print("\n=== Starting Method 2: 3 questions at a time (15 iterations) ===")
    method_2_results = []
    for i in range(15):
        print(f"  Method 2 - Iteration {i + 1}/15...")
        try:
            response = client.models.generate_content(
                model=generation_model,
                contents=method_2_prompt,
                config=config,
            )
            try:
                response_text = get_response_text(response)
                data = extract_json(response_text)
                if isinstance(data, list):
                    method_2_results.extend(data)
                else:
                    print(f"    Expected a list but got dict on iteration {i+1}")
            except json.JSONDecodeError as e:
                print(f"    Failed to parse JSON on iteration {i+1}: {e}")
                print(f"    Raw output: {response_text}")
        except Exception as e:
            print(f"    Error on iteration {i+1}: {e}")
            traceback.print_exc()
        time.sleep(2)  # avoid rate limits

    # Save raw results
    with open("raw_results.json", "w") as f:
        json.dump(
            {"method_1": method_1_results, "method_2": method_2_results}, f, indent=2
        )

    print("\n=== Evaluating Results with Gemini Flash ===")

    evaluation_prompt_template = """
You are an expert A-Level Mathematics examiner.
Evaluate the following generated exam question based on two criteria:
1. Correctness: Is the math absolutely flawless? Does the mark scheme correctly solve the question? Are the LaTeX escapes correct?
2. Quality: Is it highly original, non-trivial, and appropriately difficult?

Question Data:
{question_data}

Provide your evaluation as a JSON object:
{{
  "correctness_score": <int 1-10>,
  "quality_score": <int 1-10>,
  "comments": "<brief justification>"
}}
"""

    eval_config = types.GenerateContentConfig(
        temperature=0.2, response_mime_type="application/json"
    )

    def evaluate_questions(questions):
        scores = []
        for idx, q in enumerate(questions):
            print(f"  Evaluating question {idx + 1}/{len(questions)}...")
            prompt = evaluation_prompt_template.format(
                question_data=json.dumps(q, indent=2)
            )
            try:
                res = client.models.generate_content(
                    model=evaluation_model, contents=prompt, config=eval_config
                )
                eval_data = json.loads(res.text)
                scores.append(eval_data)
            except Exception as e:
                print(f"    Eval error: {e}")
            time.sleep(1)
        return scores

    print("Evaluating Method 1 questions...")
    m1_evals = evaluate_questions(method_1_results)
    print("Evaluating Method 2 questions...")
    m2_evals = evaluate_questions(method_2_results)

    # Compute averages
    def summarize(evals):
        if not evals:
            return {"avg_correctness": 0, "avg_quality": 0}
        avg_c = sum(e.get("correctness_score", 0) for e in evals) / len(evals)
        avg_q = sum(e.get("quality_score", 0) for e in evals) / len(evals)
        return {"avg_correctness": avg_c, "avg_quality": avg_q}

    m1_summary = summarize(m1_evals)
    m2_summary = summarize(m2_evals)

    final_report = {
        "method_1_summary": m1_summary,
        "method_2_summary": m2_summary,
        "method_1_evals": m1_evals,
        "method_2_evals": m2_evals,
    }

    with open("evaluation_report.json", "w") as f:
        json.dump(final_report, f, indent=2)

    print("\n=== FINAL COMPARISON ===")
    print(
        f"Method 1 (1 question x 45): Correctness={m1_summary['avg_correctness']:.2f}, Quality={m1_summary['avg_quality']:.2f}"
    )
    print(
        f"Method 2 (3 questions x 15): Correctness={m2_summary['avg_correctness']:.2f}, Quality={m2_summary['avg_quality']:.2f}"
    )


if __name__ == "__main__":
    main()
