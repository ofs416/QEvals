import json
import os
from google import genai
from dotenv import load_dotenv

def main():
    load_dotenv()
    client = genai.Client()
    
    # Reconstruct prompts
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
    method_2_prompt = base_prompt.replace(
        "- **Format:**\n{", 
        "- **Format:**\n[\n  {"
    ).replace(
        "}\n", 
        "  }\n]\n(Return a list of EXACTLY 3 such objects)"
    ) + "\nGenerate EXACTLY 3 questions."

    model = "gemini-2.5-flash"
    
    m1_prompt_tokens = client.models.count_tokens(model=model, contents=method_1_prompt).total_tokens
    m2_prompt_tokens = client.models.count_tokens(model=model, contents=method_2_prompt).total_tokens

    with open("raw_results.json", "r") as f:
        data = json.load(f)
    
    m1_results = data.get("method_1", [])
    m2_results = data.get("method_2", [])
    
    m1_output_text = "".join([json.dumps(r) for r in m1_results])
    m2_output_text = "".join([json.dumps(r) for r in m2_results])
    
    m1_out_tokens = client.models.count_tokens(model=model, contents=m1_output_text).total_tokens if m1_output_text else 0
    m2_out_tokens = client.models.count_tokens(model=model, contents=m2_output_text).total_tokens if m2_output_text else 0
    
    # We didn't capture the intermediate code execution tokens (which are part of the output), 
    # but we can estimate the final JSON output tokens. Code execution usually adds roughly 300-500 tokens per thought/code block.
    # Let's add an average of 400 tokens per question for the code execution overhead.
    code_exec_overhead_per_question = 400
    
    m1_out_tokens += code_exec_overhead_per_question * 45
    m2_out_tokens += code_exec_overhead_per_question * 45 # 15 iterations * 3 questions

    # Pricing per 1M tokens (Standard Gemini 1.5/2.5 Flash pricing)
    # Input: $0.075 / 1M
    # Input (Cached): $0.01875 / 1M
    # Output: $0.30 / 1M
    price_input_1m = 0.075
    price_input_cached_1m = 0.01875
    price_output_1m = 0.30

    # For Method 1:
    # 45 calls. 1st call is uncached. 44 calls are cached.
    m1_input_cost = (m1_prompt_tokens / 1_000_000 * price_input_1m) + (44 * m1_prompt_tokens / 1_000_000 * price_input_cached_1m)
    m1_output_cost = m1_out_tokens / 1_000_000 * price_output_1m
    m1_total = m1_input_cost + m1_output_cost

    # For Method 2:
    # 15 calls. 1st call is uncached. 14 calls are cached.
    m2_input_cost = (m2_prompt_tokens / 1_000_000 * price_input_1m) + (14 * m2_prompt_tokens / 1_000_000 * price_input_cached_1m)
    m2_output_cost = m2_out_tokens / 1_000_000 * price_output_1m
    m2_total = m2_input_cost + m2_output_cost

    print(f"=== Tokens ===")
    print(f"M1 Prompt: {m1_prompt_tokens} tokens")
    print(f"M2 Prompt: {m2_prompt_tokens} tokens")
    print(f"M1 Output (est w/ code exec): {m1_out_tokens} tokens")
    print(f"M2 Output (est w/ code exec): {m2_out_tokens} tokens")
    print(f"\n=== Costs (Gemini Flash) ===")
    print(f"Method 1 (45 calls): Input ${m1_input_cost:.5f} + Output ${m1_output_cost:.5f} = ${m1_total:.5f}")
    print(f"Method 2 (15 calls): Input ${m2_input_cost:.5f} + Output ${m2_output_cost:.5f} = ${m2_total:.5f}")

if __name__ == "__main__":
    main()
