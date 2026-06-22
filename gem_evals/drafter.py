import asyncio

from google.antigravity import Agent, CapabilitiesConfig, LocalAgentConfig

MODEL_LITE = "models/gemini-3.5-flash"

DRAFTER_PROMPT = """You are a creative edexcel PURE maths A-Level Mathematics examiner. Your task is to brainstorm a highly original, non-trivial PURE maths exam question.
Do not worry about the exact numbers being perfectly neat right now. Just focus on creating an interesting scenario or mathematical setup that combines multiple PURE maths syllabus topics.
Output ONLY the draft question text. Do not solve it. Do not write a mark scheme."""


async def run_drafter(topic_prompt: str) -> str:
    config = LocalAgentConfig(
        model=MODEL_LITE,
        system_instructions=DRAFTER_PROMPT,
        capabilities=CapabilitiesConfig(enable_write_tools=False),
    )
    async with Agent(config) as drafter:
        response = await drafter.chat(topic_prompt)
        return response.text


if __name__ == "__main__":
    # Example usage
    draft = asyncio.run(
        run_drafter("Draft a highly original question combining mechanics and vectors.")
    )
    print(draft)
