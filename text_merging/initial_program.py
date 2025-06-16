import json
import os
from typing import Dict, Any, List


# EVOLVE-BLOCK-START

def get_block_boundaries(block: Dict) -> Dict:
    """Calculates and adds top, bottom, left, right to a block dictionary for easier comparison."""
    # Ensure font size is consistent, checking for 'size' or 'font_size' keys.
    block['font_size'] = block.get('font_size', block.get('size', 0))
    # Calculate boundaries from YOLO-like coordinates (x, y, w, h)
    block['top'] = block['y'] - block['h'] / 2
    block['bottom'] = block['y'] + block['h'] / 2
    block['left'] = block['x'] - block['w'] / 2
    block['right'] = block['x'] + block['w'] / 2
    return block


def is_almost_equal(a: float, b: float, tolerance: float = 2.0) -> bool:
    """Checks if two float values are within a given tolerance."""
    return abs(a - b) <= tolerance


def is_touching(block1: Dict, block2: Dict) -> bool:
    """Checks if two blocks' bounding boxes are touching or overlapping."""
    # Standard bounding box intersection test.
    x_overlap = (block1['right'] >= block2['left']) and (block1['left'] <= block2['right'])
    y_overlap = (block1['bottom'] >= block2['top']) and (block1['top'] <= block2['bottom'])
    return x_overlap and y_overlap


def can_merge_blocks(block1: Dict, block2: Dict) -> bool:
    """
    Simulates your `can_merge` logic. Merging requires blocks to be touching
    and have at least one side aligned.
    """
    return (
            is_touching(block1, block2)
            and (
                    is_almost_equal(block1['left'], block2['left'])
                    or is_almost_equal(block1['right'], block2['right'])
                    or is_almost_equal(block1['top'], block2['top'])
                    or is_almost_equal(block1['bottom'], block2['bottom'])
            )
    )


def merge_two_blocks(block1: Dict, block2: Dict) -> Dict:
    """
    Simulates your `merge_text` logic to combine two blocks into one.
    """
    # 1. Determine the primary and secondary block based on your rules.
    primary, secondary = None, None
    if not is_almost_equal(block1['font_size'], block2['font_size'], tolerance=0.1):
        primary, secondary = (block1, block2) if block1['font_size'] > block2['font_size'] else (block2, block1)
    elif not is_almost_equal(block1['top'], block2['top']):
        primary, secondary = (block1, block2) if block1['top'] < block2['top'] else (block2, block1)
    else:
        primary, secondary = (block1, block2) if block1['left'] < block2['left'] else (block2, block1)

    # 2. Calculate the boundaries of the new merged box.
    min_left = min(primary['left'], secondary['left'])
    max_right = max(primary['right'], secondary['right'])
    min_top = min(primary['top'], secondary['top'])
    max_bottom = max(primary['bottom'], secondary['bottom'])

    # 3. Create the new merged block dictionary.
    new_block = {
        "text": primary['text'] + " " + secondary['text'],
        "font": primary['font'],
        "font_size": max(primary['font_size'], secondary['font_size']),
        "size": max(primary['font_size'], secondary['font_size']),  # For consistency
        "color": primary['color'],
        "x": (min_left + max_right) / 2,  # New center x
        "y": (min_top + max_bottom) / 2,  # New center y
        "w": max_right - min_left,
        "h": max_bottom - min_top,
    }
    # Return the new block with its boundaries also calculated for the next iteration.
    return get_block_boundaries(new_block)


def transform_text_data(input_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Merges text blocks using the iterative algorithm from your production code.
    """
    page_1_blocks = input_data.get("page1", [])
    if not page_1_blocks:
        return {"page1": []}

    # Pre-process all blocks to add boundary properties needed for merging logic.
    text_suggestions = [get_block_boundaries(block.copy()) for block in page_1_blocks]

    keep_merging = True
    merge_counter = 0
    while keep_merging:
        merge_counter += 1
        if merge_counter > 1000:  # Safety break.
            break

        start_over = False
        i = 0
        while i < len(text_suggestions):
            # To avoid issues with list modification, we check against a copy in the inner loop.
            j = 0
            while j < len(text_suggestions):
                if i == j:
                    j += 1
                    continue

                # Ensure indices are valid before accessing
                if i >= len(text_suggestions) or j >= len(text_suggestions):
                    break

                text_1 = text_suggestions[i]
                text_2 = text_suggestions[j]

                if can_merge_blocks(text_1, text_2):
                    # Perform the merge.
                    merged_block = merge_two_blocks(text_1, text_2)
                    text_suggestions[i] = merged_block
                    # Remove the consumed block.
                    del text_suggestions[j]

                    start_over = True
                    break  # Restart scan from the beginning.
                else:
                    j += 1

            if start_over:
                break
            else:
                i += 1

        # If a merge happened, we loop again. Otherwise, we're done.
        keep_merging = start_over

    # Clean up temporary keys before returning the final result.
    for block in text_suggestions:
        for key in ['top', 'bottom', 'left', 'right', 'font_size']:
            block.pop(key, None)

    return {"page1": text_suggestions}


# EVOLVE-BLOCK-END

def get_input_data(file_path: str) -> Dict[str, Any]:
    """Loads raw text block data from a JSON file."""
    if not os.path.exists(file_path):
        return {"page1": []}
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def main():
    """Main execution function for the evaluator."""
    input_file = 'text.json'
    test_data = get_input_data(input_file)
    result = transform_text_data(test_data)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()