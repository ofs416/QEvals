import asyncio
from drafter import run_drafter

async def main():
    prompts = [
        "Draft a highly original question combining mechanics and vectors.",
        "Draft a highly original question combining pure maths and integration.",
        "Draft a highly original question combining trigonometry and calculus.",
        "Draft a highly original question combining coordinate geometry and algebra.",
        "Draft a highly original question combining logarithms and series.",
        "Draft a highly original question combining differential equations and vectors.",
        "Draft a highly original question combining matrices and pure maths.",
        "Draft a highly original question combining probability and algebra.",
        "Draft a highly original question combining complex numbers and geometry.",
        "Draft a highly original question combining mechanics and pure maths."
    ]
    
    tasks = [run_drafter(p) for p in prompts]
    results = await asyncio.gather(*tasks)
    
    for i, res in enumerate(results):
        print(f"--- DRAFT {i+1} ---")
        print(res)
        print("\n")

if __name__ == "__main__":
    asyncio.run(main())
