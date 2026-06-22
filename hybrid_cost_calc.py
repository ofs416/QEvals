import json
import os
from google import genai
from dotenv import load_dotenv

def main():
    load_dotenv()
    client = genai.Client()
    
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

    model = "gemini-2.5-flash"
    
    prompt_tokens = client.models.count_tokens(model=model, contents=hybrid_prompt).total_tokens

    with open("hybrid_results.json", "r") as f:
        data = json.load(f)
    
    output_text = "".join([json.dumps(r) for r in data])
    
    out_tokens = client.models.count_tokens(model=model, contents=output_text).total_tokens if output_text else 0
    
    # 15 iterations * 3 questions = 45 questions of overhead.
    # Add an average of 400 tokens per question for the code execution overhead.
    out_tokens += 400 * 45 

    price_input_1m = 0.075
    price_input_cached_1m = 0.01875
    price_output_1m = 0.30

    # 15 calls. 1st call is uncached. 14 calls are cached.
    input_cost = (prompt_tokens / 1_000_000 * price_input_1m) + (14 * prompt_tokens / 1_000_000 * price_input_cached_1m)
    output_cost = out_tokens / 1_000_000 * price_output_1m
    total = input_cost + output_cost

    print(f"=== Hybrid Cost ===")
    print(f"Prompt Tokens: {prompt_tokens}")
    print(f"Estimated Output Tokens: {out_tokens}")
    print(f"Total Cost: ${total:.5f}")

if __name__ == "__main__":
    main()
