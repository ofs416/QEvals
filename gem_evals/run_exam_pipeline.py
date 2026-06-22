import asyncio

from drafter import run_drafter
from optimiser import run_optimiser
from typesetter import run_typesetter
from reviewer import run_reviewer

async def run_pipeline():
    print("Initializing agents with respective models...")

    print("\n=== Step 1: Drafting ===")
    draft_text = await run_drafter("Draft a highly original question combining mechanics and vectors.")
    print(f"Draft complete. Output Length: {len(draft_text)} chars")

    print("\n=== Step 2: Optimising ===")
    opt_text = await run_optimiser(draft_text)
    print(f"Optimisation complete. Output Length: {len(opt_text)} chars")

    print("\n=== Step 3: Typesetting ===")
    type_text = await run_typesetter(opt_text)
    print(f"Typesetting complete. Output Length: {len(type_text)} chars")

    print("\n=== Step 4: Reviewing ===")
    review_text = await run_reviewer(type_text)

    print("\n--- FINAL REVIEW SCORE ---")
    print(review_text)

if __name__ == "__main__":
    asyncio.run(run_pipeline())
