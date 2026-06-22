import asyncio
from google.antigravity import Agent, CapabilitiesConfig, LocalAgentConfig

MODEL_PRO = "models/gemini-2.5-pro"

REVIEWER_PROMPT = """You are a senior A-Level Mathematics examiner and Python programmer. Your task is to rigorously review and score the finalized exam question PDF/script provided to you.
CRITICAL INSTRUCTION: You MUST use your Python Code Execution environment (`run_command`) to deterministically check the calculations in the script/mark scheme. Do NOT do the math in your head.
Evaluate the question based on:
1. Mathematical correctness and flawlessness (verified with Python).
2. The "neatness" of the numbers and the elegance of the solution.
3. Originality and cross-topic integration.
Score the question out of 10 and provide detailed feedback on why you gave this score, along with any further suggestions for improvement."""

async def run_reviewer(script_text: str) -> str:
    config = LocalAgentConfig(
        model=MODEL_PRO,
        system_instructions=REVIEWER_PROMPT,
        capabilities=CapabilitiesConfig(enable_write_tools=True),
    )
    async with Agent(config) as reviewer:
        response = await reviewer.chat(f"Here is the finalized Python script. Please review and score the question:\n\n{script_text}")
        return response.text

if __name__ == "__main__":
    # Example usage
    sample_script = "print('Solving x^2 - 4 = 0')"
    review = asyncio.run(run_reviewer(sample_script))
    print(review)
