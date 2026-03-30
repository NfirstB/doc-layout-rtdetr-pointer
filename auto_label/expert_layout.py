"""
Step 1: Expert Model — YOLO Layout Detection

Uses the trained YOLO model for layout element detection.
PDF → render → YOLO detection → pseudo-labels (bbox + class + conf)

Text content is handled by the VLM step (it reads the image directly).
"""

import io
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import pypdfium2 as fitz
from PIL import Image

PARENT_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(PARENT_DIR))

from ultralytics import YOLO

LAYOUT_CLASSES = ["title", "header", "body", "figure", "caption", "table", "footnote", "equation"]


class ExpertLayoutDetector:
    """
    Expert model: YOLO layout detector on PDF pages.
    
    Pipeline:
        PDF page → render image (pypdfium2) → YOLO detection → pseudo-labels
    """
    
    def __init__(
        self,
        detector_path: Optional[str] = None,
        conf_threshold: float = 0.25,
        device: str = "cpu",
    ):
        if detector_path is None:
            detector_path = str(PARENT_DIR / "models" / "layout_yolov8n.pt")
        
        print(f"[ExpertLayout] Loading YOLO: {Path(detector_path).name}")
        self.detector = YOLO(detector_path)
        self.detector.to(device)
        self.conf_threshold = conf_threshold
        print("[ExpertLayout] Ready")
    
    def pdf_page_to_image(self, pdf_path: str, page_num: int, dpi: int = 150) -> Image.Image:
        """Render a PDF page to PIL Image."""
        doc = fitz.PdfDocument(pdf_path)
        if page_num >= len(doc):
            raise ValueError(f"PDF has {len(doc)} pages, page {page_num} out of range")
        page = doc[page_num]
        scale = dpi / 72.0
        pix = page.render(scale=scale).to_pil()
        doc.close()
        return pix
    
    def detect_from_pdf(
        self,
        pdf_path: str,
        page_num: int = 0,
        dpi: int = 150,
    ) -> tuple[list[dict], Image.Image]:
        """
        Run layout detection on a PDF page.
        
        Returns:
            (elements: list of pseudo-label dicts, page_image: PIL.Image)
        """
        page_image = self.pdf_page_to_image(pdf_path, page_num, dpi)
        return self.detect_from_image(page_image)
    
    def detect_from_image(self, img: Image.Image) -> tuple[list[dict], Image.Image]:
        """Run on an existing PIL Image."""
        img_w, img_h = img.size
        img_rgb = img.convert("RGB")
        
        results = self.detector(img_rgb, verbose=False, conf=self.conf_threshold)
        boxes = results[0].boxes
        
        elements = []
        for i, (box, conf, cls_id) in enumerate(zip(boxes.xyxy, boxes.conf, boxes.cls)):
            x0, y0, x1, y1 = box.cpu().numpy()
            x0 /= img_w; x1 /= img_w; y0 /= img_h; y1 /= img_h
            cls_name = LAYOUT_CLASSES[int(cls_id)] if int(cls_id) < len(LAYOUT_CLASSES) else "body"
            
            elements.append({
                "bbox": [float(x0), float(y0), float(x1), float(y1)],
                "text": "",   # VLM reads image directly for text content
                "conf": float(conf),
                "class_name": cls_name,
                "class_conf": float(conf),
                "source": "yolo_expert",
                "det_id": i,
            })
        
        # Sort by reading order (top-to-bottom, left-to-right)
        elements.sort(key=lambda e: (e["bbox"][1], e["bbox"][0]))
        for i, elem in enumerate(elements):
            elem["det_id"] = i
        
        return elements, img
    
    def get_pseudo_labels(self, elements: list[dict]) -> list[dict]:
        """Convert to standard pseudo-label format."""
        return [
            {
                "bbox": e["bbox"],
                "class_name": e["class_name"],
                "conf": e["conf"],
                "text": e.get("text", ""),
                "source": e.get("source", "yolo_expert"),
            }
            for e in elements
        ]


def visualize_elements(
    elements: list[dict],
    page_image: Image.Image,
    output_path: str,
    show_text: bool = False,
) -> None:
    """Visualize detected elements."""
    from PIL import ImageDraw, ImageFont
    
    CLASS_COLORS = {
        "title": (220, 60, 60), "header": (60, 120, 220), "body": (140, 140, 140),
        "figure": (60, 180, 100), "caption": (220, 200, 60), "table": (180, 60, 200),
        "footnote": (60, 160, 220), "equation": (200, 180, 60),
    }
    
    img = page_image.copy()
    W, H = img.size
    draw = ImageDraw.Draw(img)
    
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 11)
    except:
        font = ImageFont.load_default()
    
    for elem in elements:
        x0, y0, x1, y1 = elem["bbox"]
        px0, py0, px1, py1 = x0*W, y0*H, x1*W, y1*H
        color = CLASS_COLORS.get(elem["class_name"], (255, 255, 0))
        
        draw.rectangle([px0, py0, px1, py1], outline=color, width=2)
        
        label = f"{elem['det_id']}:{elem['class_name']} {elem['conf']:.2f}"
        lw, lh = draw.textbbox((0, 0), label, font=font)[2:]
        draw.rectangle([px0, max(0, py1-lh-4), px0+lw+4, py1], fill=color)
        draw.text((px0+2, max(0, py1-lh-2)), label, fill=(0, 0, 0), font=font)
    
    img.save(output_path)


def quick_test():
    """Test on sample PDF."""
    detector = ExpertLayoutDetector()
    
    test_pdf = "/home/node/.openclaw/workspace/new_papers/2403.06832.pdf"
    if not Path(test_pdf).exists():
        print(f"Test PDF not found: {test_pdf}")
        return
    
    elements, img = detector.detect_from_pdf(test_pdf, page_num=0, dpi=120)
    pseudo = detector.get_pseudo_labels(elements)
    
    print(f"\nDetected {len(pseudo)} elements:")
    for e in pseudo:
        bbox_str = ", ".join([f"{v:.2f}" for v in e["bbox"]])
        print(f"  {e['class_name']:10s} conf={e['conf']:.2f}  bbox=[{bbox_str}]")
    
    vis_path = PARENT_DIR / "auto_label_expert_vis.jpg"
    visualize_elements(elements, img, str(vis_path))
    print(f"\nVisualization: {vis_path}")


if __name__ == "__main__":
    quick_test()
