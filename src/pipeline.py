#!/usr/bin/env python3
"""
完整 Pipeline：PDF解析 → YOLO推理 → 阅读顺序推理
使用方法:
  python -m src.pipeline --pdf ./sample.pdf --output ./output/
  python -m src.pipeline --pdf-dir ./pdfs/ --output ./output/
"""
import argparse
import os
import json
import base64
import time
import sys
from pathlib import Path
from typing import List, Dict
from concurrent.futures import ThreadPoolExecutor, as_completed

# 添加 src 到路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.pdf_parser import PDFParser, TextBlock, ImageBlock, TableBlock
from src.reading_order import LayoutElement, ReadingOrderInferencer, elements_to_json


class LayoutAnalyzer:
    """完整布局分析 Pipeline"""

    def __init__(self, model_path: str = None):
        self.model = None
        self.model_path = model_path
        if model_path and os.path.exists(model_path):
            self._load_model(model_path)

    def _load_model(self, model_path: str):
        """加载 YOLO 模型"""
        try:
            from ultralytics import YOLO
            self.model = YOLO(model_path)
            print(f"  ✓ 模型已加载: {model_path}")
        except Exception as e:
            print(f"  ⚠ 模型加载失败: {e}，将使用规则方法")

    def analyze_pdf(self, pdf_path: str, output_dir: str) -> Dict:
        """分析单个 PDF"""
        os.makedirs(output_dir, exist_ok=True)

        result = {
            "pdf_path": pdf_path,
            "num_pages": 0,
            "pages": [],
            "metadata": {},
        }

        start = time.time()
        with PDFParser(pdf_path) as parser:
            pages = parser.parse_all()
            result["num_pages"] = len(pages)

            for page_info in pages:
                page_result = self._analyze_page(page_info, parser)
                result["pages"].append(page_result)

                # 保存页面渲染图（带检测框）
                self._render_page_with_boxes(page_info, page_result, output_dir)

        result["analysis_time"] = time.time() - start

        # 保存 JSON 结果
        json_path = os.path.join(output_dir, "layout_result.json")
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        print(f"  ✓ 结果已保存: {json_path}")

        return result

    def _analyze_page(self, page_info, parser) -> Dict:
        """分析单页"""
        page_result = {
            "page_num": page_info.page_num,
            "width": page_info.width,
            "height": page_info.height,
            "reading_order": [],
            "stats": {},
        }

        # 构建布局元素
        elements: List[LayoutElement] = []

        # 添加文本块
        for i, tb in enumerate(page_info.texts):
            elem = LayoutElement(
                id=f"t_{page_info.page_num}_{i}",
                elem_type=tb.block_type,
                page=page_info.page_num,
                x0=tb.x0, y0=tb.y0, x1=tb.x1, y1=tb.y1,
                text=tb.text,
            )
            elements.append(elem)

        # 添加图片块
        for i, ib in enumerate(page_info.images):
            elem = LayoutElement(
                id=f"i_{page_info.page_num}_{i}",
                elem_type=ib.block_type,
                page=page_info.page_num,
                x0=ib.x0, y0=ib.y0, x1=ib.x1, y1=ib.y1,
            )
            elements.append(elem)

        # 添加表格块
        for i, tb in enumerate(page_info.tables):
            elem = LayoutElement(
                id=f"tbl_{page_info.page_num}_{i}",
                elem_type=tb.block_type,
                page=page_info.page_num,
                x0=tb.x0, y0=tb.y0, x1=tb.x1, y1=tb.y1,
            )
            elements.append(elem)

        # 如果有 YOLO 模型，用模型检测补充
        if self.model:
            elements = self._detect_with_model(page_info, elements, parser)

        # 推理阅读顺序
        infer = ReadingOrderInferencer(page_info.width, page_info.height)
        sorted_elements = infer.infer(elements)

        # 转换为 JSON
        page_result["reading_order"] = elements_to_json(sorted_elements)

        # 统计
        type_counts = {}
        for e in sorted_elements:
            type_counts[e.elem_type] = type_counts.get(e.elem_type, 0) + 1
        page_result["stats"] = type_counts

        return page_result

    def _detect_with_model(self, page_info, elements: List[LayoutElement], parser) -> List[LayoutElement]:
        """用 YOLO 模型检测布局元素"""
        try:
            png_bytes = parser.render_page_to_image(page_info.page_num, dpi=150)
            results = self.model(png_bytes, verbose=False)

            for r in results:
                boxes = r.boxes
                if boxes is not None:
                    for box in boxes:
                        cls_id = int(box.cls[0].item())
                        cls_name = self.model.names[cls_id]
                        xyxy = box.xyxy[0].tolist()
                        conf = float(box.conf[0].item())

                        elem = LayoutElement(
                            id=f"y_{page_info.page_num}_{cls_id}_{len(elements)}",
                            elem_type=cls_name,
                            page=page_info.page_num,
                            x0=xyxy[0], y0=xyxy[1],
                            x1=xyxy[2], y1=xyxy[3],
                            confidence=conf,
                        )
                        elements.append(elem)
        except Exception as e:
            print(f"  ⚠ YOLO 检测失败: {e}")

        return elements

    def _render_page_with_boxes(self, page_info, page_result: Dict, output_dir: str):
        """渲染页面并绘制检测框和阅读顺序（可选项）"""
        # 渲染功能需要 pdf_path，后续扩展
        pass


def analyze_batch(pdf_dir: str, output_dir: str, model_path: str = None, max_workers: int = 4):
    """批量分析"""
    analyzer = LayoutAnalyzer(model_path)
    pdf_files = list(Path(pdf_dir).glob("*.pdf"))
    print(f"找到 {len(pdf_files)} 个 PDF 文件")

    os.makedirs(output_dir, exist_ok=True)

    def process_one(pdf_path):
        name = Path(pdf_path).stem
        out = os.path.join(output_dir, name)
        print(f"\n处理: {pdf_path}")
        try:
            return analyzer.analyze_pdf(str(pdf_path), out)
        except Exception as e:
            print(f"  ✗ 失败: {e}")
            return None

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(process_one, str(p)): p for p in pdf_files}
        results = []
        for future in as_completed(futures):
            results.append(future.result())

    success = sum(1 for r in results if r is not None)
    print(f"\n=== 完成: {success}/{len(pdf_files)} 成功 ===")


def main():
    parser = argparse.ArgumentParser(description="文档布局分析 Pipeline")
    parser.add_argument("--pdf", type=str, help="单个 PDF 文件")
    parser.add_argument("--pdf-dir", type=str, help="PDF 目录")
    parser.add_argument("--output", type=str, default="./output", help="输出目录")
    parser.add_argument("--model", type=str, default=None, help="YOLO 模型路径")
    parser.add_argument("--workers", type=int, default=4, help="并发数")
    args = parser.parse_args()

    if args.pdf:
        analyzer = LayoutAnalyzer(args.model)
        analyzer.analyze_pdf(args.pdf, args.output)
    elif args.pdf_dir:
        analyze_batch(args.pdf_dir, args.output, args.model, args.workers)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
