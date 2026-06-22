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
        raise ValueError("Empty text")
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

    drafter_prompt = """You are a creative A-Level Mathematics examiner. Your task is to brainstorm a highly original, non-trivial exam question.
Do not worry about the exact numbers being perfectly neat right now. Just focus on creating an interesting scenario or mathematical setup that combines multiple syllabus topics (e.g., calculus and trigonometry, mechanics and vectors, or probability and series).

Output ONLY the draft question text. Do not solve it. Do not write a mark scheme.
"""

    refiner_prompt_template = """You are an expert A-Level Mathematics examiner and Python programmer. You have been given a draft exam question. 
Your task is to finalize this question and create a mathematically flawless, detailed mark scheme.

DRAFT QUESTION:
{draft_question}

INSTRUCTIONS:
1. Try to solve the draft question using your Python Code Execution environment.
2. If the final answers or intermediate steps result in "ugly" numbers (e.g., nasty decimals, un-factorable quadratics, unhelpful angles), you MUST tweak the constants or givens in the original question to make the numbers "neat" (e.g., integers, simple fractions, clean surds like \\\\sqrt{{3}}, exact multiples of \\\\pi). Use Python to reverse-engineer these clean constants.
3. Once you have a perfectly working question with neat numbers, output the finalized question and the step-by-step mark scheme.

OUTPUT FORMAT:
Provide your final output in a strict JSON format inside a markdown block.

- **JSON/LaTeX Escaping [CRITICAL]:** Every LaTeX backslash must be written as a DOUBLE backslash inside JSON strings.
  - Correct: "Find $\\\\frac{{dy}}{{dx}}$"

```json
{{
  "topic": "String",
  "difficulty": "higher" | "extension",
  "question_latex": "The finalized exam question formatted in LaTeX.",
  "markscheme_latex": "The step-by-step solution formatted in LaTeX.",
  "verification_summary": "Brief summary of what constants you tweaked to make the numbers nice."
}}
```
"""

    model_name = "gemini-2.5-flash"

    drafter_config = types.GenerateContentConfig(
        temperature=0.9, # Higher temperature for creativity
    )
    
    refiner_config = types.GenerateContentConfig(
        temperature=0.4, # Lower temperature for analytical rigor
        tools=[{"code_execution": {}}],
    )

    print("=== Starting Draft-Refine Pipeline (10 questions) ===")
    results = []
    
    for i in range(10):
        print(f"\nIteration {i+1}/10...")
        try:
            # 1. DRAFT
            print("  -> Drafting concept...")
            draft_res = client.models.generate_content(
                model=model_name,
                contents=drafter_prompt,
                config=drafter_config,
            )
            draft_text = get_response_text(draft_res)
            
            # 2. REFINE & SOLVE
            print("  -> Refining, executing code, and finalizing...")
            refine_prompt = refiner_prompt_template.format(draft_question=draft_text)
            refine_res = client.models.generate_content(
                model=model_name,
                contents=refine_prompt,
                config=refiner_config,
            )
            refine_text = get_response_text(refine_res)
            
            # 3. EXTRACT
            try:
                data = extract_json(refine_text)
                # Store the original draft for comparison
                data["original_draft"] = draft_text.strip()
                results.append(data)
                print(f"  -> Success! Tweaked: {data.get('verification_summary', 'None')}")
            except Exception as e:
                print(f"  -> Failed to parse JSON: {e}")
                
        except Exception as e:
            print(f"  -> Pipeline Error: {e}")
            traceback.print_exc()
        
        time.sleep(2)

    # Save raw results
    with open("draft_refine_results.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nSuccessfully generated {len(results)} questions.")
    print("\n=== Evaluating Draft-Refine Results with Gemini Flash ===")
    
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

    evals = []
    for idx, q in enumerate(results):
        print(f"  Evaluating question {idx+1}/{len(results)}...")
        prompt = evaluation_prompt_template.format(question_data=json.dumps(q, indent=2))
        try:
            res = client.models.generate_content(
                model=model_name,
                contents=prompt,
                config=eval_config
            )
            eval_data = json.loads(res.text)
            evals.append(eval_data)
        except Exception as e:
            print(f"    Eval error: {e}")
        time.sleep(1)

    # Compute averages
    if evals:
        avg_c = sum(e.get("correctness_score", 0) for e in evals) / len(evals)
        avg_q = sum(e.get("quality_score", 0) for e in evals) / len(evals)
    else:
        avg_c, avg_q = 0, 0

    final_report = {
        "draft_refine_summary": {"avg_correctness": avg_c, "avg_quality": avg_q},
        "evals": evals
    }

    with open("draft_refine_evaluation_report.json", "w") as f:
        json.dump(final_report, f, indent=2)

    print("\n=== FINAL DRAFT-REFINE SCORE ===")
    print(f"Draft-Refine Method (10 questions): Correctness={avg_c:.2f}, Quality={avg_q:.2f}")

if __name__ == "__main__":
    main()
