"""
Step 3: Hallucination Filtering

Filters out false positive detections from the VLM refinement step.
Uses multi-level filtering:
1. Rule-based geometric/positional checks
2. Class-specific heuristics
3. Text content validation
4. Optional: VLM-based hallucination scoring
"""

from typing import Optional

import numpy as np

# ── Class-specific rules ──────────────────────────────────────────────────────

CLASS_RULES = {
    "title": {
        "min_y_ratio": 0.0,
        "max_y_ratio": 0.20,   # Must be in top 20% of page
        "max_area_ratio": 0.7, # Can't cover more than 70% of page
        "min_height_ratio": 0.01,
        "max_height_ratio": 0.10,
        "reject_keywords": [],  # Keywords that suggest this is NOT a title
    },
    "header": {
        "min_y_ratio": 0.0,
        "max_y_ratio": 0.15,
        "max_area_ratio": 0.5,
        "min_height_ratio": 0.005,
        "max_height_ratio": 0.08,
    },
    "body": {
        "min_y_ratio": 0.10,   # Below title/header area
        "max_y_ratio": 0.95,
        "min_height_ratio": 0.03,
        "max_height_ratio": 0.70,
    },
    "figure": {
        "min_height_ratio": 0.05,
        "min_width_ratio": 0.08,
        "max_area_ratio": 0.8,
    },
    "caption": {
        "min_y_ratio": 0.05,
        "max_height_ratio": 0.15,
        "max_text_len": 500,
    },
    "table": {
        "min_height_ratio": 0.05,
        "min_width_ratio": 0.10,
        "max_area_ratio": 0.8,
    },
    "footnote": {
        "min_y_ratio": 0.80,   # Must be in bottom 20%
        "max_y_ratio": 1.0,
        "max_height_ratio": 0.15,
        "max_text_len": 300,
    },
    "equation": {
        "max_y_ratio": 0.95,
        "max_height_ratio": 0.25,
        "max_text_len": 500,
    },
}


class HallucinationFilter:
    """
    Filters out hallucinated layout element detections.
    
    Strategy:
    1. Class-specific geometric rules (position, size)
    2. Text content validation (is there actual text for text classes?)
    3. Confidence threshold
    4. Intra-element consistency checks
    """
    
    def __init__(
        self,
        conf_threshold: float = 0.3,
        use_strict_rules: bool = True,
    ):
        """
        Args:
            conf_threshold: Minimum confidence to keep an element
            use_strict_rules: If True, reject elements that violate class rules
        """
        self.conf_threshold = conf_threshold
        self.use_strict_rules = use_strict_rules
    
    def filter(self, elements: list[dict]) -> tuple[list[dict], list[dict]]:
        """
        Filter hallucinated elements.
        
        Returns:
            (kept_elements, removed_elements)
        """
        kept = []
        removed = []
        
        for elem in elements:
            is_hallucinated, reason = self._check_element(elem, elements)
            
            if is_hallucinated:
                elem_copy = elem.copy()
                elem_copy["removed_reason"] = reason
                removed.append(elem_copy)
            else:
                kept.append(elem)
        
        return kept, removed
    
    def _check_element(self, elem: dict, all_elements: list[dict]) -> tuple[bool, Optional[str]]:
        """
        Check if an element is a hallucination.
        
        Returns:
            (is_hallucination: bool, reason: str or None)
        """
        class_name = elem.get("class_name", "body")
        bbox = elem.get("bbox", [0, 0, 1, 1])
        conf = elem.get("conf", 1.0)
        text = elem.get("text", "")
        x0, y0, x1, y1 = bbox
        
        page_h = 1.0
        page_w = 1.0
        area = (x1 - x0) * (y1 - y0)
        height = y1 - y0
        width = x1 - x0
        
        # Rule 1: Confidence threshold
        if conf < self.conf_threshold:
            return True, f"confidence {conf:.2f} below threshold {self.conf_threshold}"
        
        # Rule 2: Geometric rules for this class
        rules = CLASS_RULES.get(class_name, {})
        
        if rules:
            if "min_y_ratio" in rules and y0 < rules["min_y_ratio"]:
                return True, f"y0={y0:.2f} below min_y_ratio={rules['min_y_ratio']}"
            if "max_y_ratio" in rules and y1 > rules["max_y_ratio"]:
                return True, f"y1={y1:.2f} above max_y_ratio={rules['max_y_ratio']}"
            if "min_height_ratio" in rules and height < rules["min_height_ratio"]:
                return True, f"height={height:.3f} below min={rules['min_height_ratio']}"
            if "max_height_ratio" in rules and height > rules["max_height_ratio"]:
                return True, f"height={height:.3f} above max={rules['max_height_ratio']}"
            if "min_width_ratio" in rules and width < rules["min_width_ratio"]:
                return True, f"width={width:.3f} below min={rules['min_width_ratio']}"
            if "max_area_ratio" in rules and area > rules["max_area_ratio"]:
                return True, f"area={area:.3f} above max={rules['max_area_ratio']}"
            if "max_text_len" in rules and len(text) > rules["max_text_len"]:
                return True, f"text_len={len(text)} exceeds max={rules['max_text_len']}"
        
        # Rule 3: Class-specific content checks
        if class_name in ("title", "header", "footnote", "equation"):
            # These should have actual text
            if not text or len(text.strip()) < 3:
                if class_name in ("title", "header"):
                    return True, f"{class_name} has no text content"
        
        if class_name == "title":
            # Title shouldn't appear in bottom half of page
            if y0 > 0.5:
                return True, "title appearing in bottom half of page (suspicious)"
        
        if class_name == "figure":
            # Figure should be reasonably sized
            if area < 0.01:
                return True, "figure area too small"
            # Figure usually has text or visual content
            if not text and conf < 0.7:
                return True, "figure has no caption and low confidence"
        
        if class_name == "table":
            # Table detection without table-specific indicators
            if not text and conf < 0.5:
                return True, "table has no text and low confidence"
        
        # Rule 4: Check for duplicate/overlapping elements
        for other in all_elements:
            if other is elem or other.get("class_name") != class_name:
                continue
            ob = other.get("bbox", [0,0,0,0])
            if (bbox[0] < ob[2] and bbox[2] > ob[0] and 
                bbox[1] < ob[3] and bbox[3] > ob[1]):
                # Overlapping - keep the higher confidence one
                if conf < other.get("conf", 1.0):
                    return True, f"overlapping with higher conf element (conf={conf:.2f})"
        
        # Rule 5: Negative keywords
        if class_name in ("title", "header"):
            lower_text = text.lower()
            if any(kw in lower_text for kw in ["http://", "https://", "www.", ".com"]):
                return True, "title/header contains URL"
        
        # All checks passed
        return False, None
    
    def filter_by_text_density(self, elements: list[dict]) -> list[dict]:
        """
        Additional pass: filter text classes that have suspiciously empty content.
        """
        kept = []
        for elem in elements:
            cls = elem.get("class_name", "")
            text = elem.get("text", "")
            
            # Text classes should have some text
            if cls in ("body", "footnote", "caption"):
                if not text.strip() and elem.get("conf", 1.0) < 0.8:
                    elem_copy = elem.copy()
                    elem_copy["removed_reason"] = "no text content for text class"
                    continue
            
            kept.append(elem)
        
        return kept


def summarize_filter_results(kept: list[dict], removed: list[dict]) -> str:
    """Generate a human-readable summary of filtering results."""
    lines = [
        f"Total elements before filter: {len(kept) + len(removed)}",
        f"Kept: {len(kept)}",
        f"Removed: {len(removed)}",
        "",
    ]
    
    if removed:
        by_reason: dict[str, int] = {}
        for r in removed:
            reason = r.get("removed_reason", "unknown")
            by_reason[reason] = by_reason.get(reason, 0) + 1
        
        lines.append("Removal reasons:")
        for reason, count in sorted(by_reason.items(), key=lambda x: -x[1]):
            lines.append(f"  {reason}: {count}")
    
    by_class: dict[str, int] = {}
    for e in kept:
        cls = e.get("class_name", "?")
        by_class[cls] = by_class.get(cls, 0) + 1
    
    lines.append("")
    lines.append("Kept elements by class:")
    for cls, count in sorted(by_class.items()):
        lines.append(f"  {cls}: {count}")
    
    return "\n".join(lines)
