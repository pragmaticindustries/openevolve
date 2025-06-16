import json
import os
from typing import Dict, Any, List

def get_block_boundaries(block: Dict) -> Dict:
    """Adds boundary info to a block: top, bottom, left, right, font_size."""
    b = block.copy()
    b['font_size'] = b.get('font_size', b.get('size', 0))
    b['top'] = b['y'] - b['h'] / 2
    b['bottom'] = b['y'] + b['h'] / 2
    b['left'] = b['x'] - b['w'] / 2
    b['right'] = b['x'] + b['w'] / 2
    return b

def looks_like_headline(block: Dict) -> bool:
    text = block.get('text', '').strip()
    font_size = block.get('font_size', 0)
    if not text:
        return False
    if font_size >= 20:
        return True
    if text and text[0] in {'#', '-', '*', '\u2022', '>', '='}:
        return True
    if len(text) < 40 and font_size >= 16:
        return True
    return False

def blocks_are_close(b1: Dict, b2: Dict, gap_tol: float = 8.0) -> bool:
    # Consider blocks adjacent if they overlap horizontally and their vertical gap is small, or vice versa
    h_overlap = not (b1['right'] < b2['left'] or b2['right'] < b1['left'])
    v_overlap = not (b1['bottom'] < b2['top'] or b2['bottom'] < b1['top'])
    v_gap = min(abs(b1['top'] - b2['bottom']), abs(b2['top'] - b1['bottom']))
    h_gap = min(abs(b1['left'] - b2['right']), abs(b2['left'] - b1['right']))
    # Prefer vertical stacking
    if h_overlap and v_gap <= gap_tol:
        return True
    if v_overlap and h_gap <= gap_tol:
        return True
    return False

def pick_headline_first(b1: Dict, b2: Dict) -> List[Dict]:
    h1 = looks_like_headline(b1)
    h2 = looks_like_headline(b2)
    if h1 and not h2:
        return [b1, b2]
    if h2 and not h1:
        return [b2, b1]
    if b1['font_size'] > b2['font_size']:
        return [b1, b2]
    if b2['font_size'] > b1['font_size']:
        return [b2, b1]
    # If both are similar, preserve upper (top) first
    if b1['top'] < b2['top']:
        return [b1, b2]
    return [b2, b1]

def merge_blocks(b1: Dict, b2: Dict) -> Dict:
    # Merge two blocks, headline first, and update all geometry
    [first, second] = pick_headline_first(b1, b2)
    merged_text = (first['text'].strip() + " " + second['text'].strip()).strip()
    font = first.get('font', second.get('font', ''))
    color = first.get('color', second.get('color', ''))
    max_font_size = max(first['font_size'], second['font_size'])
    min_left = min(first['left'], second['left'])
    max_right = max(first['right'], second['right'])
    min_top = min(first['top'], second['top'])
    max_bottom = max(first['bottom'], second['bottom'])
    x = (min_left + max_right) / 2
    y = (min_top + max_bottom) / 2
    w = max_right - min_left
    h = max_bottom - min_top
    merged_block = {
        "text": merged_text,
        "font": font,
        "font_size": max_font_size,
        "size": max_font_size,
        "color": color,
        "x": x,
        "y": y,
        "w": w,
        "h": h
    }
    return get_block_boundaries(merged_block)

def transform_text_data(input_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Merges adjacent text blocks, keeping headlines first in merged text.
    """
    blocks = input_data.get("page1", [])
    if not blocks:
        return {"page1": []}
    # Add boundaries for all blocks up front
    items = [get_block_boundaries(b) for b in blocks]

    # Use a greedy approach: always merge the closest pair until no more can be merged
    changed = True
    max_iters = 500
    iters = 0
    while changed and iters < max_iters:
        changed = False
        n = len(items)
        used = [False] * n
        merged_indices = set()
        new_items = []
        i = 0
        while i < n:
            if used[i]:
                i += 1
                continue
            best_j = -1
            for j in range(n):
                if i == j or used[j]:
                    continue
                if blocks_are_close(items[i], items[j]):
                    best_j = j
                    break
            if best_j != -1:
                merged_block = merge_blocks(items[i], items[best_j])
                used[i] = True
                used[best_j] = True
                new_items.append(merged_block)
                changed = True
            else:
                new_items.append(items[i])
                used[i] = True
            i += 1
        items = new_items
        iters += 1

    # Remove temporary keys
    for b in items:
        for k in ['top', 'bottom', 'left', 'right', 'font_size']:
            if k in b:
                b.pop(k)
    return {"page1": items}

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
