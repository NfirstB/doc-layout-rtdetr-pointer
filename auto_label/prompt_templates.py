"""
Prompt Engineering Templates for VLM Refinement
"""

REFINE_SYSTEM_PROMPT = """You are an expert at analyzing scientific paper pages.
Your task is to refine pseudo-labels (layout elements detected by an OCR model) for this document page.

For each detected element, you must:
1. Verify if it actually exists in the image
2. Correct the bounding box if it's misaligned
3. Fix the class label if it's wrong (title/header/body/figure/caption/table/footnote/equation)
4. Keep the element if it's correct

Rules:
- title: Paper title, chapter title, or section heading (usually at top of page or section)
- header: Page header, journal name, author info at top
- body: Main text paragraphs
- figure: Charts, diagrams, photographs
- caption: Text describing a figure (usually directly below a figure)
- table: Tabular data with rows and columns
- footnote: Notes at bottom of page
- equation: Standalone mathematical formulas or equations

IMPORTANT: Only output valid JSON. No explanations outside JSON."""

REFINE_USER_PROMPT_TEMPLATE = """Here is a page from a scientific paper.

The OCR model detected {n_elements} layout elements. Here are the pseudo-labels:

{pseudo_labels}

Study the image carefully and refine each element.
- Keep bbox coordinates as [x0, y0, x1, y1] normalized to [0, 1]
- Fix any wrong class labels
- Remove any hallucinated elements that don't exist
- Add any missing important elements
- Set confidence to 1.0 if you're certain, lower if uncertain

Output a JSON array of elements with this exact format:
[
  {{
    "bbox": [x0, y0, x1, y1],
    "class_name": "title|header|body|figure|caption|table|footnote|equation",
    "conf": 0.0-1.0,
    "corrected": true|false
  }},
  ...
]

Be precise about bounding box coordinates - they must match the actual image content."""

def build_refine_prompt(pseudo_labels: list[dict], image_base64: str = None) -> dict:
    """
    Build the refine prompt for VLM.
    
    Args:
        pseudo_labels: List of element dicts from expert model
        image_base64: Base64 encoded image (not used in prompt, passed via API)
    
    Returns:
        dict with messages for API call
    """
    # Format pseudo-labels for display
    labels_text = []
    for i, elem in enumerate(pseudo_labels):
        labels_text.append(
            f"  Element {i+1}: bbox={elem.get('bbox')}, "
            f"class={elem.get('class_name')}, conf={elem.get('conf', 0):.2f}, "
            f"text={elem.get('text', '')[:50]!r}"
        )
    labels_str = "\n".join(labels_text)
    
    user_prompt = REFINE_USER_PROMPT_TEMPLATE.format(
        n_elements=len(pseudo_labels),
        pseudo_labels=labels_str
    )
    
    return {
        "system": REFINE_SYSTEM_PROMPT,
        "user": user_prompt
    }


HALLUCINATION_CHECK_PROMPT = """You are checking whether detected layout elements are hallucinated.
For each element, determine if it actually exists in the image.

Class-specific hallucination rules:
- title: Only at top ~15% of page or at section starts. If near bottom, likely hallucinated.
- figure: Should have meaningful visual content inside bbox. If bbox is mostly white/blank, likely false positive.
- body: Should contain text lines. If bbox covers image-only area with no text, likely wrong.
- table: Should have visible horizontal/vertical lines or structured text. If not, check carefully.
- equation: Should contain mathematical symbols or clearly look like a formula.

For each element, output JSON:
{{
  "elem_idx": 0,
  "is_hallucination": true|false,
  "reason": "explanation if hallucinated"
}}"""
