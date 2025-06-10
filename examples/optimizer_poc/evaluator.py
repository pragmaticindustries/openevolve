"""
Evaluator for sort algorithm
"""

import importlib.util
import numpy as np
import time
import os
import signal
import subprocess
import tempfile
import traceback
import sys
import pickle


class TimeoutError(Exception):
    pass


def timeout_handler(signum, frame):
    """Handle timeout signal"""
    raise TimeoutError("Function execution timed out")


def validate_best_node(best_node_costs, n):
    """
    Validate that the list is sorted in ascending order

    Args:
        sorted_list: List of numbers to validate, either python float or numpy array

    Returns:
        True if valid, False otherwise
    """
    if best_node_costs is None:
        raise ValueError("Best node costs cannot be None, this means no solution was found")
    # This is for n = 5
    if n == 5:
        # 3200
        # 640
        return best_node_costs == [600, 300, 4000.0, 150.0, 10500]
    # This is for n = 10
    if n == 10:
        # 7280
        # 728
        return best_node_costs == [1050, 800, 9000.0, 300.0, 21000]
    if n == 12:
        # 8090
        # 675
        return best_node_costs == [1260, 900, 10000.0, 200.0, 25200]
    if n == 15:
        # 10520
        # 701
        # 8900
        return best_node_costs == [1590, 1200, 13000.0, 200.0, 31800]
    return False


def run_with_timeout(program_path, n, timeout_seconds=120):
    """
    Run the program in a separate process with timeout
    using a simple subprocess approach

    Args:
        program_path: Path to the program file
        timeout_seconds: Maximum execution time in seconds

    Returns:
        centers, radii, sum_radii tuple from the program
    """
    # Create a temporary file to execute
    with tempfile.NamedTemporaryFile(suffix=".py", delete=False) as temp_file:
        # Write a script that executes the program and saves results
        script = f"""
import sys
import numpy as np
import os
import pickle
import traceback

# Add the directory to sys.path
sys.path.insert(0, os.path.dirname('{program_path}'))

# Debugging info
print(f"Running in subprocess, Python version: {{sys.version}}")
print(f"Program path: {program_path}")

try:
    # Import the program
    spec = __import__('importlib.util').util.spec_from_file_location("program", '{program_path}')
    program = __import__('importlib.util').util.module_from_spec(spec)
    spec.loader.exec_module(program)

    # Run the packing function
    print("Calling run_sorting()...")
    beste_node_costs = program.run_sorting({n})
    # print(f"run_packing() returned successfully: sum_radii = {{sum_radii}}")

    # Save results to a file
    results = {{
        'beste_node_costs': beste_node_costs
    }}

    with open('{temp_file.name}.results', 'wb') as f:
        pickle.dump(results, f)
    print(f"Results saved to {temp_file.name}.results")

except Exception as e:
    # If an error occurs, save the error instead
    print(f"Error in subprocess: {{str(e)}}")
    traceback.print_exc()
    with open('{temp_file.name}.results', 'wb') as f:
        pickle.dump({{'error': str(e)}}, f)
    print(f"Error saved to {temp_file.name}.results")
"""
        temp_file.write(script.encode())
        temp_file_path = temp_file.name

    results_path = f"{temp_file_path}.results"

    try:
        # Run the script with timeout
        process = subprocess.Popen(
            [sys.executable, temp_file_path], stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )

        try:
            stdout, stderr = process.communicate(timeout=timeout_seconds)
            exit_code = process.returncode

            # Always print output for debugging purposes
            print(f"Subprocess stdout: {stdout.decode()}")
            if stderr:
                print(f"Subprocess stderr: {stderr.decode()}")

            # Still raise an error for non-zero exit codes, but only after printing the output
            if exit_code != 0:
                raise RuntimeError(f"Process exited with code {exit_code}")

            # Load the results
            if os.path.exists(results_path):
                with open(results_path, "rb") as f:
                    results = pickle.load(f)

                # Check if an error was returned
                if "error" in results:
                    raise RuntimeError(f"Program execution failed: {results['error']}")

                return results["beste_node_costs"]
            else:
                raise RuntimeError("Results file not found")

        except subprocess.TimeoutExpired:
            # Kill the process if it times out
            process.kill()
            process.wait()
            raise TimeoutError(f"Process timed out after {timeout_seconds} seconds")

    finally:
        # Clean up temporary files
        if os.path.exists(temp_file_path):
            os.unlink(temp_file_path)
        if os.path.exists(results_path):
            os.unlink(results_path)


def evaluate(program_path):
    """
    Evaluate the program by running it once and checking the sum of radii

    Args:
        program_path: Path to the program file

    Returns:
        Dictionary of metrics
    """
    # Target value from the paper
    # TARGET_VALUE = 2.635  # AlphaEvolve result for n=26

    try:
        # For constructor-based approaches, a single evaluation is sufficient
        # since the result is deterministic
        start_time = time.time()

        for n in [12]:

            # Use subprocess to run with timeout
            sorted_list = run_with_timeout(
                program_path, n, timeout_seconds=600  # Single timeout
            )

            # Validate solution
            valid = validate_best_node(sorted_list, n)

            # TODO
            # Approximate the result as good as possible with weights [0.0, 0.1, 0.8, 0.0, 0.0]

            # multiply the result by weights (for 15)
            max_value = 10520

            weights = [0.0, 0.1, 0.8, 0.0, 0.0]

            weighted_result = sum(
                weight * value for weight, value in zip(weights, sorted_list)
            )

            if not valid:
                break

        end_time = time.time()
        eval_time = end_time - start_time

        print(
            f"Evaluation: valid={valid}, time={eval_time:.2f}s"
        )

        # if not valid:
        #     raise ValueError(f"Invalid solution")

        return {
            # "eval_time": 1.0 / float(eval_time),
            # "weighted_result": weighted_result/max_value,
            "approximate_optimal": (n * 650)/weighted_result
        }

    except Exception as e:
        print(f"Evaluation failed completely: {str(e)}")
        traceback.print_exc()
        return {
            # "eval_time": 0.0,
            # "weighted_result": 0.0,
            "approximate_optimal": 0.0
        }
