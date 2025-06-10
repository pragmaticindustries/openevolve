# EVOLVE-BLOCK-START
"""Constructor-based circle packing for n=26 circles"""
import builtins

import numpy as np


def construct_sorting(initial_list):
    """
    Initialize the list of 1000 entries with random integers and sort them
    """
    # Sort the list using a simple sorting algorithn

    # Outer loop to iterate through the list n times
    for n in range(len(initial_list) - 1, 0, -1):

        # Initialize swapped to track if any swaps occur
        swapped = False

        # Inner loop to compare adjacent elements
        for i in range(n):
            if initial_list[i] > initial_list[i + 1]:
                # Swap elements if they are in the wrong order
                initial_list[i], initial_list[i + 1] = initial_list[i + 1], initial_list[i]

                # Mark that a swap has occurred
                swapped = True

        # If no swaps occurred, the list is already sorted
        if not swapped:
            break

    return initial_list


# EVOLVE-BLOCK-END


# This part remains fixed (not evolved)
def run_sorting():
    """Run the circle packing constructor for n=26"""

    n = 100000

    # to be comparable we need to store the list in a file and load it on all consecutive runs
    # check if the file exists, if not create it
    # get path of this file

    try:
        initial_list = np.load('/tmp/initial_list.npy')
        print("Loaded initial list from file.")
    except FileNotFoundError:
        print("Initial list file not found, generating a new one.")
        # Generate a list of n random integers between 0 and 1000
        initial_list = np.random.randint(0, high=100000, size=n)

        # store the initial list in a file
        np.save('/tmp/initial_list.npy', initial_list)

        print("Initial list generated and saved to file.")

    def error_handler(*args, **kwargs):
        raise RuntimeError("This function is forbidden to be used and should not be called.")

    np.ma.core.sort = error_handler
    # np.ndarray.sort = error_handler
    np.sort = error_handler
    builtins.sort = error_handler

    sorted_list = construct_sorting(initial_list)
    return sorted_list


if __name__ == "__main__":
    sorted_list = run_sorting()
    print("List was sorted successfully.")
