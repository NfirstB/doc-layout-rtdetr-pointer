"""
Main Auto-Label Pipeline

Three-step automatic labeling:
  1. Expert Model (PaddleOCR)      → Pseudo-labels
  2. VLM Refinement (MiniMax VL)  → Corrected labels  
  3. Hallucination Filter          → Clean high-quality labels

Usage:
    from auto_label import AutoLabelPipeline
    
    pipeline = AutoLabelPipeline()
    result = pipeline.run("/path/to/paper.pdf", page=0)
    
    print(result["elements"])   # Final labeled elements
    print(result["stats"])      # Statistics
    print(result["kept"])       # Elements after filtering
    print(result["removed"])    # Removed hallucinated elements
"""

import io
import json
import time
from pathlib import Path
from typing import Optional

from PIL import Image

from .expert_layout import ExpertLayoutDetector
from .vlm_refine import VLMRefiner
from .filter_hallucination import HallucinationFilter, summarize_filter_results


class AutoLabelPipeline:
    """
    Automatic labeling pipeline: Expert → VLM → Filter
    """
    
    def __init__(
        self,
        expert_conf_threshold: float = 0.5,
        filter_conf_threshold: float = 0.3,
        skip_vlm: bool = False,
        skip_filter: bool = False,
    ):
        """
        Args:
            expert_conf_threshold: Minimum OCR confidence for pseudo-labels
            filter_conf_threshold: Minimum confidence after filtering
            skip_vlm: Skip VLM refinement (for debugging)
            skip_filter: Skip hallucination filtering (for debugging)
        """
        self.skip_vlm = skip_vlm
        self.skip_filter = skip_filter
        
        print("[Pipeline] Initializing AutoLabel pipeline...")
        self.expert = ExpertLayoutDetector(
            lang="en",
            use_angle_cls=True,
            ocr_conf_threshold=expert_conf_threshold,
        )
        
        if not skip_vlm:
            self.vlm = VLMRefiner()
        
        if not skip_filter:
            self.filter = HallucinationFilter(conf_threshold=filter_conf_threshold)
        
        print("[Pipeline] Initialization complete")
    
    def run(
        self,
        pdf_path: str,
        page_num: int = 0,
        dpi: int = 150,
        output_dir: Optional[str] = None,
    ) -> dict:
        """
        Run the full automatic labeling pipeline.
        
        Args:
            pdf_path: Path to PDF file
            page_num: Page number to process (0-indexed)
            dpi: DPI for PDF rendering
            output_dir: Optional directory to save results
        
        Returns:
            dict with keys:
                - elements: List of labeled elements
                - pseudo_labels: Raw pseudo-labels from expert
                - refined_labels: Labels after VLM refinement
                - kept: Elements kept after filtering
                - removed: Elements removed by filter
                - stats: Summary statistics
                - page_image: PIL Image of the page
        """
        t0 = time.time()
        pdf_path = str(pdf_path)
        page_info = {"pdf_path": pdf_path, "page_num": page_num, "dpi": dpi}
        
        print(f"\n[Pipeline] Processing: {Path(pdf_path).name} page {page_num+1}")
        print("=" * 60)
        
        # ── Step 1: Expert Layout Detection ─────────────────────────────────
        t1 = time.time()
        print(f"\n[Step 1/3] Expert layout detection (PaddleOCR)...")
        
        elements, page_image = self.expert.detect_from_pdf(pdf_path, page_num, dpi)
        pseudo_labels = self.expert.get_pseudo_labels(elements)
        
        print(f"  Generated {len(pseudo_labels)} pseudo-labels in {time.time()-t1:.1f}s")
        
        if output_dir:
            self._save_pseudo_labels(pseudo_labels, output_dir, pdf_path, page_num)
        
        # ── Step 2: VLM Refinement ──────────────────────────────────────────
        if self.skip_vlm:
            refined_labels = pseudo_labels
            print(f"\n[Step 2/3] Skipped (--skip_vlm)")
        else:
            t2 = time.time()
            print(f"\n[Step 2/3] VLM refinement (MiniMax VL)...")
            
            img_bytes = io.BytesIO()
            page_image.save(img_bytes, format="JPEG", quality=90)
            img_bytes = img_bytes.getvalue()
            
            try:
                refined_labels = self.vlm.refine(
                    image=img_bytes,
                    pseudo_labels=pseudo_labels,
                    page_info=page_info,
                )
                print(f"  Refined to {len(refined_labels)} elements in {time.time()-t2:.1f}s")
            except Exception as e:
                print(f"  VLM refinement failed: {e}")
                print("  Falling back to pseudo-labels with reduced confidence")
                refined_labels = pseudo_labels
        
        # ── Step 3: Hallucination Filtering ────────────────────────────────
        if self.skip_filter:
            kept = refined_labels
            removed = []
            print(f"\n[Step 3/3] Skipped (--skip_filter)")
        else:
            t3 = time.time()
            print(f"\n[Step 3/3] Hallucination filtering...")
            
            kept, removed = self.filter.filter(refined_labels)
            
            # Second pass: text density check
            kept = self.filter.filter_by_text_density(kept)
            
            print(f"  Kept: {len(kept)}, Removed: {len(removed)} in {time.time()-t3:.1f}s")
        
        # ── Summary ────────────────────────────────────────────────────────
        total_time = time.time() - t0
        
        stats = {
            "total_time_seconds": round(total_time, 1),
            "pseudo_labels_count": len(pseudo_labels),
            "refined_count": len(refined_labels),
            "kept_count": len(kept),
            "removed_count": len(removed),
            "by_class": self._count_by_class(kept),
            "filter_summary": summarize_filter_results(kept, removed) if not self.skip_filter else "",
        }
        
        print(f"\n[Pipeline] Complete in {total_time:.1f}s")
        print(f"  Pseudo-labels: {len(pseudo_labels)}")
        print(f"  After VLM:     {len(refined_labels)}")
        print(f"  After filter:  {len(kept)} kept, {len(removed)} removed")
        
        return {
            "elements": kept,
            "pseudo_labels": pseudo_labels,
            "refined_labels": refined_labels,
            "kept": kept,
            "removed": removed,
            "stats": stats,
            "page_image": page_image,
            "page_info": page_info,
        }
    
    def run_batch(
        self,
        pdf_paths: list[str],
        pages_per_pdf: Optional[list[int]] = None,
        dpi: int = 150,
        output_dir: Optional[str] = None,
        progress_callback=None,
    ) -> list[dict]:
        """
        Process multiple PDFs.
        
        Args:
            pdf_paths: List of PDF file paths
            pages_per_pdf: List of page numbers (one per PDF), or None for page 0
            dpi: DPI for rendering
            output_dir: Directory to save per-page JSON results
            progress_callback: Optional callback(current, total) for progress updates
        """
        if pages_per_pdf is None:
            pages_per_pdf = [0] * len(pdf_paths)
        
        results = []
        total = len(pdf_paths)
        
        for i, (pdf_path, page_num) in enumerate(zip(pdf_paths, pages_per_pdf)):
            print(f"\n{'='*60}")
            print(f"[Batch] {i+1}/{total}: {Path(pdf_path).name}")
            
            result = self.run(pdf_path, page_num, dpi, output_dir)
            results.append(result)
            
            if progress_callback:
                progress_callback(i + 1, total)
            
            # Rate limiting between VLM calls
            if i < total - 1:
                time.sleep(1)
        
        return results
    
    def _save_pseudo_labels(
        self,
        pseudo_labels: list[dict],
        output_dir: str,
        pdf_path: str,
        page_num: int,
    ) -> None:
        """Save pseudo-labels to JSON file."""
        out_path = Path(output_dir) / f"{Path(pdf_path).stem}_p{page_num}_pseudo.json"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump({
                "source": "paddleocr_expert",
                "pdf_path": pdf_path,
                "page_num": page_num,
                "elements": pseudo_labels,
            }, f, ensure_ascii=False, indent=2)
        
        print(f"  Saved pseudo-labels: {out_path}")
    
    def _count_by_class(self, elements: list[dict]) -> dict[str, int]:
        counts = {}
        for e in elements:
            cls = e.get("class_name", "?")
            counts[cls] = counts.get(cls, 0) + 1
        return counts


def main():
    """CLI entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Auto Label Pipeline")
    parser.add_argument("pdf", help="PDF file path")
    parser.add_argument("--page", type=int, default=0, help="Page number (0-indexed)")
    parser.add_argument("--dpi", type=int, default=150, help="PDF rendering DPI")
    parser.add_argument("--output", "-o", help="Output directory")
    parser.add_argument("--skip-vlm", action="store_true", help="Skip VLM refinement")
    parser.add_argument("--skip-filter", action="store_true", help="Skip hallucination filter")
    
    args = parser.parse_args()
    
    pipeline = AutoLabelPipeline(
        skip_vlm=args.skip_vlm,
        skip_filter=args.skip_filter,
    )
    
    result = pipeline.run(args.pdf, page_num=args.page, dpi=args.dpi, output_dir=args.output)
    
    print("\n" + "=" * 60)
    print("FINAL LABELED ELEMENTS:")
    for i, elem in enumerate(result["kept"]):
        print(f"  {i+1}. {elem.get('class_name'):10s} "
              f"conf={elem.get('conf', 0):.2f}  "
              f"bbox={[f'{v:.2f}' for v in elem.get('bbox',[])]}  "
              f"text={elem.get('text','')[:50]!r}")
    
    # Save final result
    if args.output:
        out_path = Path(args.output) / f"{Path(args.pdf).stem}_p{args.page}_labels.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump({
                "page_info": result["page_info"],
                "stats": result["stats"],
                "elements": result["elements"],
                "pseudo_labels": result["pseudo_labels"],
                "removed": result["removed"],
            }, f, ensure_ascii=False, indent=2)
        print(f"\nSaved to: {out_path}")


if __name__ == "__main__":
    main()
