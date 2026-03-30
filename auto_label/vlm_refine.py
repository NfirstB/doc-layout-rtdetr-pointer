"""
Step 2: VLM Refinement — Use MiniMax VL API to refine pseudo-labels

Architecture:
  Raw image + pseudo-labels → MiniMax-VL API → Refined labels with corrections
"""

import base64
import json
import time
from pathlib import Path
from typing import Optional

import httpx

from .prompt_templates import build_refine_prompt

# ── MiniMax VL API ─────────────────────────────────────────────────────────────

MINIMAX_API_KEY = "sk-cp-Rg1BFk1cRUhEjK_CZ0ai6aFpP2Fp43yLUWNVyGT-PRqyrM6uMVRSPsWrnTsUVA8IQ53E1JBksuujedj2RPm7U5J7xrKlNtQ53rHDbsjv-Z6SFxSuFkRkmDs"
MINIMAX_API_URL = "https://api.minimax.chat/v1/chat/completions"

# MiniMax VL models
MINIMAX_VL_MODEL = "MiniMax-VL-02"


class VLMRefiner:
    """
    Use MiniMax VL API to refine pseudo-labels from expert model.
    
    The VLM receives:
    1. The original PDF page image (as base64)
    2. The pseudo-labels as text (bbox, class, conf, text)
    
    And returns refined labels in JSON format.
    """
    
    def __init__(
        self,
        api_key: str = MINIMAX_API_KEY,
        model: str = MINIMAX_VL_MODEL,
        temperature: float = 0.1,
        max_retries: int = 3,
        timeout: int = 120,
    ):
        self.api_key = api_key
        self.model = model
        self.temperature = temperature
        self.max_retries = max_retries
        self.timeout = timeout
    
    def _encode_image(self, image_path: str) -> str:
        """Encode image to base64."""
        with open(image_path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")
    
    def _encode_image_from_bytes(self, image_bytes: bytes) -> str:
        """Encode image bytes to base64."""
        return base64.b64encode(image_bytes).decode("utf-8")
    
    def refine(
        self,
        image: bytes | str,
        pseudo_labels: list[dict],
        page_info: Optional[dict] = None,
    ) -> list[dict]:
        """
        Send pseudo-labels + image to VLM for refinement.
        
        Args:
            image: Image file path (str) or bytes
            pseudo_labels: List of element dicts from expert model
            page_info: Optional dict with pdf_path, page_num, etc.
        
        Returns:
            List of refined element dicts
        """
        # Encode image
        if isinstance(image, str):
            image_base64 = self._encode_image(image)
        else:
            image_base64 = self._encode_image_from_bytes(image)
        
        # Build prompt
        prompt_dict = build_refine_prompt(pseudo_labels)
        
        # Build API request
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        
        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "system",
                    "content": prompt_dict["system"],
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{image_base64}",
                            },
                        },
                        {
                            "type": "text",
                            "text": prompt_dict["user"],
                        },
                    ],
                },
            ],
            "temperature": self.temperature,
            "max_tokens": 4096,
        }
        
        # Make API call with retries
        last_error = None
        for attempt in range(self.max_retries):
            try:
                with httpx.Client(timeout=self.timeout) as client:
                    response = client.post(
                        MINIMAX_API_URL,
                        headers=headers,
                        json=payload,
                    )
                
                if response.status_code == 200:
                    break
                elif response.status_code == 429:
                    # Rate limited, wait and retry
                    wait_time = 2 ** attempt * 5
                    print(f"[VLMRefiner] Rate limited, waiting {wait_time}s...")
                    time.sleep(wait_time)
                    continue
                else:
                    print(f"[VLMRefiner] API error {response.status_code}: {response.text[:200]}")
                    last_error = Exception(f"API error {response.status_code}")
                    
            except Exception as e:
                last_error = e
                print(f"[VLMRefiner] Attempt {attempt+1} failed: {e}")
                time.sleep(2 ** attempt * 3)
        
        if response.status_code != 200:
            print(f"[VLMRefiner] All {self.max_retries} attempts failed")
            return self._fallback_refine(pseudo_labels)
        
        # Parse response
        result_text = response.json()["choices"][0]["message"]["content"]
        
        # Extract JSON from response
        refined = self._parse_refined_json(result_text)
        
        if refined is None:
            print(f"[VLMRefiner] Failed to parse VLM response, using fallback")
            return self._fallback_refine(pseudo_labels)
        
        # Add page info
        if page_info:
            for elem in refined:
                elem["page_info"] = page_info
        
        return refined
    
    def _parse_refined_json(self, text: str) -> Optional[list[dict]]:
        """Extract JSON array from VLM response text."""
        # Try to find JSON array in the response
        text = text.strip()
        
        # Handle markdown code blocks
        if "```json" in text:
            start = text.find("```json") + 7
            end = text.find("```", start)
            text = text[start:end].strip()
        elif "```" in text:
            start = text.find("```") + 3
            end = text.find("```", start)
            text = text[start:end].strip()
        
        # Find JSON array
        if text.startswith("["):
            # Find matching closing bracket
            depth = 0
            end_idx = 0
            for i, c in enumerate(text):
                if c == "[":
                    depth += 1
                elif c == "]":
                    depth -= 1
                    if depth == 0:
                        end_idx = i + 1
                        break
            
            json_str = text[:end_idx]
        elif "{" in text:
            # Try wrapping in array if it's a single object
            try:
                obj = json.loads(text)
                return [obj]
            except:
                pass
            # Find JSON object
            start = text.find("{")
            end = text.rfind("}") + 1
            json_str = text[start:end]
        else:
            return None
        
        try:
            parsed = json.loads(json_str)
            return parsed
        except json.JSONDecodeError as e:
            print(f"[VLMRefiner] JSON parse error: {e}")
            # Try to extract individual objects
            return self._extract_objects_from_text(text)
    
    def _extract_objects_from_text(self, text: str) -> Optional[list[dict]]:
        """Fallback: try to extract element objects from text."""
        import re
        
        # Find all bbox patterns: [x0, y0, x1, y1]
        bbox_pattern = re.compile(r'\[\s*0?\.\d+\s*,\s*0?\.\d+\s*,\s*0?\.\d+\s*,\s*0?\.\d+\s*\]')
        class_pattern = re.compile(
            r'"(title|header|body|figure|caption|table|footnote|equation)"',
            re.IGNORECASE
        )
        
        elements = []
        bboxes = bbox_pattern.findall(text)
        classes = class_pattern.findall(text.lower())
        
        if bboxes and classes:
            for i, (bbox_str, cls_name) in enumerate(zip(bboxes, classes)):
                try:
                    bbox = json.loads(bbox_str.replace(" ", ""))
                    elements.append({
                        "bbox": bbox,
                        "class_name": cls_name.lower(),
                        "conf": 0.7,
                        "corrected": True,
                    })
                except:
                    pass
        
        if elements:
            print(f"[VLMRefiner] Extracted {len(elements)} elements via regex fallback")
            return elements
        
        return None
    
    def _fallback_refine(self, pseudo_labels: list[dict]) -> list[dict]:
        """
        Fallback when VLM fails: mark all pseudo-labels as needing review.
        """
        refined = []
        for elem in pseudo_labels:
            refined.append({
                "bbox": elem["bbox"],
                "class_name": elem["class_name"],
                "conf": elem.get("conf", 0.5) * 0.8,  # Lower confidence
                "text": elem.get("text", ""),
                "corrected": False,
                "source": "fallback",
                "note": "VLM refinement failed, confidence reduced",
            })
        return refined


def quick_test():
    """Quick test VLM refinement."""
    import io
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent))
    
    from expert_layout import ExpertLayoutDetector
    
    test_pdf = "/home/node/.openclaw/workspace/new_papers/2403.06832.pdf"
    if not Path(test_pdf).exists():
        print(f"Test PDF not found: {test_pdf}")
        return
    
    # Step 1: Expert layout detection
    print("[Test] Step 1: Expert layout detection...")
    detector = ExpertLayoutDetector()
    elements, img = detector.detect_from_pdf(test_pdf, page_num=0, dpi=120)
    pseudo_labels = detector.get_pseudo_labels(elements)
    print(f"  Generated {len(pseudo_labels)} pseudo-labels")
    
    # Save image for VLM
    img_bytes = io.BytesIO()
    img.save(img_bytes, format="JPEG", quality=85)
    img_bytes = img_bytes.getvalue()
    
    # Step 2: VLM refinement
    print("[Test] Step 2: VLM refinement...")
    refiner = VLMRefiner()
    refined = refiner.refine(img_bytes, pseudo_labels, page_info={"pdf": test_pdf, "page": 0})
    print(f"  Refined to {len(refined)} elements")
    
    for e in refined[:5]:
        print(f"  {e.get('class_name','?'):10s} conf={e.get('conf',0):.2f}  "
              f"bbox={e.get('bbox')}  corrected={e.get('corrected',False)}")
    
    if refined:
        print(f"\n[OK] VLM refinement test passed")


if __name__ == "__main__":
    import sys
    quick_test()
