import os
import json
import time
import traceback
from google import genai
from google.genai import types
from dotenv import load_dotenv

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

def main():
    load_dotenv()
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print("Error: Please set the GEMINI_API_KEY environment variable.")
        return

    client = genai.Client(api_key=api_key)

    hybrid_prompt = """You are an expert A-Level Mathematics examiner, pedagogical specialist, and Python programmer. Your task is to generate mathematically flawless, highly original exam questions and detailed mark schemes.

You have access to a Python Code Execution environment. You MUST use it to act as a deterministic oracle and verify the mathematical soundness of your question BEFORE outputting the final response.

### THE WORK-BACKWARDS PROTOCOL (MANDATORY)
1. **Answer First:** Do not write the question text first. Choose the final, clean, "nice" answers first (e.g., integers, simple fractions, clear surds).
2. **Reverse Engineer:** Write Python code (using `sympy`, `numpy`, or `math`) to calculate the required constants, coefficients, or givens that lead to your chosen answers.
3. **Execute & Verify:** Run the code. Check for unintended edge cases:
   - Are there multiple roots/intersections when only one is expected?
   - Does a calculus derivative simplify so much that the question loses its intended difficulty?
   - Are physical/statistical constraints violated (e.g., negative friction, empty statistical critical regions)?
4. **Finalize:** Only after the Python execution confirms the math is perfect and the values are sound, construct the final question text and mark scheme.

### CRITICAL RULE FOR BATCH GENERATION
You must generate exactly 3 questions. However, you MUST perform steps 1-4 (including writing and running an isolated Python script) for the FIRST question, completely verifying it, BEFORE you begin working on the SECOND question. 
Do NOT try to verify all 3 questions in a single Python script. Isolate them completely. 

### DOMAIN GUARDRAILS
- **Pure Math:** Verify multi-step integrals, geometric series sums, and ensure derivatives don't collapse to trivial linear results.
- **Mechanics:** Ensure Normal Reaction R >= 0, static friction <= mu*R.
- **Statistics:** Verify all critical values and probabilities using Python. Never hand-sum expected values.

### OUTPUT SCHEMA & CRITICAL RULES
Respond ONLY with a valid JSON array of objects. No prose outside the JSON blocks.

- **JSON/LaTeX Escaping [CRITICAL]:** Every LaTeX backslash must be written as a DOUBLE backslash inside JSON strings.
  - Correct: "Find $\\frac{dy}{dx}$"
  - Wrong: "Find $\\frac{dy}{dx}$"
- **Format:**
[
  {
    "topic": "String",
    "difficulty": "foundation" | "higher" | "extension",
    "question_latex": "The exam question formatted in LaTeX using $ for inline and $$ for display.",
    "markscheme_latex": "The step-by-step solution formatted in LaTeX. Use double backslashes for all commands like \\\\int or \\\\frac.",
    "verification_summary": "A brief 1-sentence explanation of what you reverse-engineered and verified with Python."
  }
]
(Return a list of EXACTLY 3 such objects)
"""

    generation_model = "gemini-2.5-flash"
    evaluation_model = "gemini-2.5-flash"

    config = types.GenerateContentConfig(
        temperature=0.7,
        tools=[{"code_execution": {}}],
    )

    print("=== Starting Hybrid Method: 3 questions at a time with isolated code execution (15 iterations) ===")
    hybrid_results = []
    for i in range(15):
        print(f"  Hybrid - Iteration {i+1}/15...")
        try:
            response = client.models.generate_content(
                model=generation_model,
                contents=hybrid_prompt,
                config=config,
            )
            try:
                response_text = get_response_text(response)
                data = extract_json(response_text)
                if isinstance(data, list):
                    hybrid_results.extend(data)
                else:
                    print(f"    Expected a list but got dict on iteration {i+1}")
            except json.JSONDecodeError as e:
                print(f"    Failed to parse JSON on iteration {i+1}: {e}")
                print(f"    Raw output (truncated): {response_text[:200]}...")
        except Exception as e:
            print(f"    Error on iteration {i+1}: {e}")
            traceback.print_exc()
        time.sleep(2)  # avoid rate limits

    # Save raw results
    with open("hybrid_results.json", "w") as f:
        json.dump(hybrid_results, f, indent=2)

    print("\n=== Evaluating Hybrid Results with Gemini Flash ===")
    
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
        temperature=0.2,
        response_mime_type="application/json"
    )

    def evaluate_questions(questions):
        scores = []
        for idx, q in enumerate(questions):
            print(f"  Evaluating question {idx+1}/{len(questions)}...")
            prompt = evaluation_prompt_template.format(question_data=json.dumps(q, indent=2))
            try:
                res = client.models.generate_content(
                    model=evaluation_model,
                    contents=prompt,
                    config=eval_config
                )
                eval_data = json.loads(res.text)
                scores.append(eval_data)
            except Exception as e:
                print(f"    Eval error: {e}")
            time.sleep(1)
        return scores

    hybrid_evals = evaluate_questions(hybrid_results)

    # Compute averages
    if hybrid_evals:
        avg_c = sum(e.get("correctness_score", 0) for e in hybrid_evals) / len(hybrid_evals)
        avg_q = sum(e.get("quality_score", 0) for e in hybrid_evals) / len(hybrid_evals)
    else:
        avg_c, avg_q = 0, 0

    final_report = {
        "hybrid_summary": {"avg_correctness": avg_c, "avg_quality": avg_q},
        "hybrid_evals": hybrid_evals
    }

    with open("hybrid_evaluation_report.json", "w") as f:
        json.dump(final_report, f, indent=2)

    print("\n=== FINAL HYBRID SCORE ===")
    print(f"Hybrid Method (3 questions x 15): Correctness={avg_c:.2f}, Quality={avg_q:.2f}")

if __name__ == "__main__":
    main()
