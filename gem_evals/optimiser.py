import asyncio
from google.antigravity import Agent, CapabilitiesConfig, LocalAgentConfig

MODEL_LITE = "models/gemini-3.5-flash"

OPTIMISER_PROMPT = """You are an expert A-Level Mathematics problem solver and Python programmer. You have been given a draft exam question.
Your task is solely to finalize the mathematics of this question and guarantee correctness, specifically by working backwards.
INSTRUCTIONS:
1. CRITICAL: Start by defining a "nice", elegant final solution/answer and work completely backwards step-by-step using your Python Code Execution environment via the `sympy` library.
2. By working backwards from the desired neat result, determine what the initial constants and parameters in the question must be to produce that exact neat result. Double-check this rigorously with Python.
3. Determine the logical steps required for the mark scheme.
OUTPUT FORMAT:
Output the finalized Question text, and crucially, output the **exact Python `sympy` code snippet** you successfully used to verify the numbers. Do NOT write a Python script for rendering, and do NOT worry about LaTeX formatting. An example is given in evals\\gem_evals\\render.py"""

async def run_optimiser(draft_text: str) -> str:
    config = LocalAgentConfig(
        model=MODEL_LITE,
        system_instructions=OPTIMISER_PROMPT,
        capabilities=CapabilitiesConfig(enable_write_tools=True),
    )
    async with Agent(config) as optimiser:
        response = await optimiser.chat(f"Here is the draft question. Please finalize the mathematics:\n\n{draft_text}")
        return response.text

if __name__ == "__main__":
    # Example usage
    sample_draft = "Draft question about a particle moving with constant acceleration a = 3i - 2j m/s^2."
    opt = asyncio.run(run_optimiser(sample_draft))
    print(opt)
