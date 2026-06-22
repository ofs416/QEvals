import asyncio
from google.antigravity import Agent, CapabilitiesConfig, LocalAgentConfig

MODEL_LITE = "models/gemini-3.5-flash"

TYPESETTER_PROMPT = """You are an expert Python developer and LaTeX typesetter. You have been given the finalized text and the verified `sympy` Python code for an exam question.
Your task is to wrap this into a final executable rendering script.
INSTRUCTIONS:
1. Your final output must be a **single standalone Python script**.
2. CRITICAL: You MUST use the exact `sympy` code provided by the Optimizer to perform the calculations. Do NOT invent new math logic.
3. Define the final Question and Mark Scheme as a large Python `f-string` template containing LaTeX.
4. Dynamically inject the computed `sympy` variables from the Optimizer's code into the f-string using `{sp.latex(variable)}`.
5. Use `sympy.preview` (or similar) to deterministically render the final PDF.
OUTPUT FORMAT:
Output ONLY the comprehensive Python script. Do NOT output raw markdown text."""

async def run_typesetter(optimised_text: str) -> str:
    config = LocalAgentConfig(
        model=MODEL_LITE,
        system_instructions=TYPESETTER_PROMPT,
        capabilities=CapabilitiesConfig(enable_write_tools=True),
    )
    async with Agent(config) as typesetter:
        response = await typesetter.chat(f"Here is the finalized math and sympy code. Please create the Python rendering script:\n\n{optimised_text}")
        return response.text

if __name__ == "__main__":
    # Example usage
    sample_opt = "Question: Find the root of x^2 - 4 = 0. \\n Sympy Code: import sympy as sp; x = sp.Symbol('x'); print(sp.solve(x**2 - 4, x))"
    type_script = asyncio.run(run_typesetter(sample_opt))
    print(type_script)
