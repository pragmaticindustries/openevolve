from openevolve import OpenEvolve

# Initialize the system
evolve = OpenEvolve(
    initial_program_path="examples/heiko/simulate_orders.py",
    # initial_program_path="examples/optimizer_poc/initial_program_5.py",
    evaluation_file="examples/heiko/evaluator.py",
    config_path="examples/heiko/config.yaml",
)

evolve.database.load("examples/heiko/openevolve_output/checkpoints/checkpoint_45")  # Load from checkpoint if available

import asyncio

# Run the evolution
async def main():
    best_program = await evolve.run(iterations=1000)
    print(f"Best program metrics:")
    for name, value in best_program.metrics.items():
        print(f"  {name}: {value:.4f}")

asyncio.run(main())