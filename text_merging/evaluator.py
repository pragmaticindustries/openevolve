import json
import subprocess
import time


def get_ground_truth():
    """
    Loads the expected output from the ground truth JSON file.
    """
    # The ground truth file must be named 'text_result.json' in the same directory.
    with open('text_result.json', 'r', encoding='utf-8') as f:
        return json.load(f)


def evaluate(program_path: str):
    """
    Evaluates an evolved program by running it and checking if its output
    contains the expected text blocks from the ground truth.
    """
    print(f"\n--- Starting evaluation for {program_path} ---")
    try:
        start_time = time.time()
        # Ensure the input file 'text.json' is available for the program to use.
        result = subprocess.run(
            ['python3', program_path],
            capture_output=True,
            text=True,
            timeout=99999  # Increased timeout for potentially complex merging
        )
        eval_time = time.time() - start_time
        print(f"Evaluation took {eval_time:.4f} seconds.")

        if result.returncode != 0:
            error_message = f"Execution failed with return code {result.returncode}. Stderr: {result.stderr}"
            print(f"ERROR: {error_message}")
            return {"correctness": 0.0, "inverse_eval_time": 0.0, "error": error_message}

        # Print the raw output from the program for debugging
        print("--- Program Output (stdout) ---")
        print(result.stdout)
        print("-----------------------------")

        try:
            output_data = json.loads(result.stdout)
        except json.JSONDecodeError as e:
            error_message = f"Output was not valid JSON. Error: {e}"
            print(f"ERROR: {error_message}")
            return {"correctness": 0.0, "inverse_eval_time": 0.0, "error": error_message}

        ground_truth = get_ground_truth()

        output_blocks = output_data.get("page1", [])
        ground_truth_blocks = ground_truth.get("page1", [])

        if not ground_truth_blocks:
            correctness = 100.0 if not output_blocks else 0.0
            print("Ground truth is empty. Correctness is 100% if output is also empty.")
            return {"correctness": correctness, "inverse_eval_time": 1.0 / eval_time if eval_time > 0 else 0.0}

        # Create a set of all unique text strings from the program's output for fast checking.
        output_texts = {block.get('text', '').strip() for block in output_blocks}
        print(f"\nFound {len(output_texts)} unique text blocks in output.")

        score = 0
        unmatched_texts = []

        print("\n--- Comparing with Ground Truth ---")
        # Count how many of the ground truth texts are present in the output.
        for truth_block in ground_truth_blocks:
            expected_text = truth_block.get('text', '').strip()
            if expected_text and expected_text in output_texts:
                score += 1
                print(f"[  OK   ] Found: \"{expected_text}\"")
            elif expected_text:
                unmatched_texts.append(expected_text)
                print(f"[ MISS  ] Did not find: \"{expected_text}\"")

        if unmatched_texts:
            print("\n--- Summary of Missing Texts ---")
            for text in unmatched_texts:
                print(f"- \"{text}\"")
            print("--------------------------------")


        # Calculate correctness as the percentage of matched texts.
        max_score = len(ground_truth_blocks)
        correctness = (score / max_score) * 100.0 if max_score > 0 else 0.0

        print(f"\n--- Final Score ---")
        print(f"Matched {score} out of {max_score} ground truth blocks.")
        print(f"Correctness: {correctness:.2f}%")
        print("-------------------")

        return {"correctness": correctness, "inverse_eval_time": 1.0 / eval_time if eval_time > 0 else 0.0}

    except subprocess.TimeoutExpired:
        print("ERROR: Evaluation timed out.")
        return {"correctness": 0.0, "inverse_eval_time": 0.0, "error": "Evaluation timed out."}
    except Exception as e:
        print(f"ERROR: An unexpected error occurred: {e}")
        return {"correctness": 0.0, "inverse_eval_time": 0.0, "error": str(e)}


if __name__ == "__main__":
    # This block allows you to run the evaluator directly for testing.
    # It will evaluate the 'initial_program.py' by default.
    # The OpenEvolve framework will call the evaluate() function directly,
    # so this part won't run in that context.
    print(">>> Running evaluator in standalone mode for debugging <<<")
    # Make sure the path to the program you want to test is correct.
    program_to_evaluate = 'text_merging/initial_program.py'
    evaluate(program_to_evaluate)
