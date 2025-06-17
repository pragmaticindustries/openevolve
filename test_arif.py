from openevolve import OpenEvolve

# Initialize the system
evolve = OpenEvolve(
    initial_program_path="text_merging/initial_program.py",
    # initial_program_path="examples/optimizer_poc/initial_program_5.py",
    evaluation_file="text_merging/evaluator.py",
    config_path="text_merging/config.yaml",
)

evolve.database.load("text_merging/openevolve_output2/checkpoints/checkpoint_170")  # Load from checkpoint if available

import asyncio

# Run the evolution
async def main():
    best_program = await evolve.run(iterations=1000)
    print(f"Best program metrics:")
    for name, value in best_program.metrics.items():
        print(f"  {name}: {value:.4f}")

asyncio.run(main())