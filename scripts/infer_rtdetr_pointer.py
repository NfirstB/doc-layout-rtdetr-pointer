#!/usr/bin/env python3
"""
RT-DETR + Pointer Network 完整推理 pipeline

架构：
  1. RT-DETR（或YOLO）：检测文档元素，输出边界框 + 类别
  2. Pointer Network：预测阅读顺序

用法：
  python scripts/infer_rtdetr_pointer.py \
      --detector models/layout_yolov8n.pt \
      --pointer models/pointer_network.pt \
      --pdf paper.pdf
"""
import argparse
import io
import sys
from pathlib import Path
import json

import torch
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from pdf_parser import PDFParser
from ultralytics import YOLO

from pointer_network import (
    ReadingOrderModel,
    generate_pseudo_order,
)


# ── 类别映射 ─────────────────────────────────────────────────────────────────

CLASS_NAMES = ["title", "header", "body", "figure", "caption", "table", "footnote", "equation"]
CLASS_COLORS = {
    "title": (220, 60, 60),
    "header": (60, 120, 220),
    "body": (140, 140, 140),
    "figure": (60, 180, 100),
    "caption": (220, 200, 60),
    "table": (180, 60, 200),
    "footnote": (60, 160, 220),
    "equation": (200, 180, 60),
}
NUM_CLASSES = len(CLASS_NAMES)


# ── 主推理流程 ───────────────────────────────────────────────────────────────

def detect_elements(model_detector, img_pil, conf_threshold=0.3):
    """
    用检测模型找出所有元素

    Returns:
        bboxes: (N, 4) 归一化 [x0, y0, x1, y1]
        categories: (N,)
        confidences: (N,)
    """
    results = model_detector(img_pil, verbose=False, conf=conf_threshold)
    boxes = results[0].boxes

    if len(boxes) == 0:
        return np.array([]), np.array([]), np.array([])

    # 提取边界框
    xyxy = boxes.xyxy.cpu().numpy()  # (N, 4)
    confs = boxes.conf.cpu().numpy()
    cls_ids = boxes.cls.cpu().numpy().astype(int)

    # 归一化到 [0, 1]
    W, H = img_pil.size
    bboxes = xyxy.copy()
    bboxes[:, [0, 2]] /= W  # x0, x1
    bboxes[:, [1, 3]] /= H  # y0, y1

    return bboxes, cls_ids, confs


def predict_order(pointer_model, bboxes, categories):
    """
    用指针网络预测阅读顺序

    Returns:
        order: List[int]，元素索引顺序
    """
    if len(bboxes) == 0:
        return []

    if pointer_model is not None:
        order = pointer_model.predict(bboxes, categories)
    else:
        # Fallback: 启发式
        order = generate_pseudo_order(bboxes)

    return order


def visualize(img_pil, bboxes, categories, confidences, order,
              output_path="output.jpg"):
    """
    可视化检测结果和阅读顺序（透明框，不挡文字）
    """
    W, H = img_pil.size

    # 直接在 RGBA 模式的图片上绘制（避免 alpha_composite 色彩问题）
    img = img_pil.convert("RGBA")
    draw = ImageDraw.Draw(img, "RGBA")

    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 13)
        font_bold = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 18)
    except:
        font = ImageDraw.font.load_default()
        font_bold = font

    # 建立 idx → order position 映射
    order_pos = {elem_idx: pos for pos, elem_idx in enumerate(order)}

    # 画检测框（半透明填充 + 实线边框）
    for i, (bbox, cat_id, conf) in enumerate(zip(bboxes, categories, confidences)):
        x0, y0, x1, y1 = bbox
        px0, py0, px1, py1 = x0 * W, y0 * H, x1 * W, y1 * H

        cls_name = CLASS_NAMES[cat_id] if cat_id < len(CLASS_NAMES) else "unknown"
        color = CLASS_COLORS.get(cls_name, (255, 255, 0))

        # 半透明填充 + 边框（不挡文字）
        draw.rectangle([px0, py0, px1, py1],
                       fill=color + (50,),    # 半透明
                       outline=color + (220,), # 带 alpha
                       width=2)

        # 标签放在框外下方（不遮挡内容）
        label = f"{cls_name} {conf:.2f}"
        lw, lh = draw.textbbox((0, 0), label, font=font)[2:]
        draw.rectangle([px0, py1 + 2, px0 + lw + 8, py1 + lh + 8],
                       fill=color + (180,), outline=color + (255,), width=1)
        draw.text((px0 + 4, py1 + 4), label, fill=(0, 0, 0, 255), font=font)

    # 画阅读顺序（黄点 + 数字，放在框内中心）
    for elem_idx in order:
        bbox = bboxes[elem_idx]
        x0, y0, x1, y1 = bbox
        cx = (x0 + x1) / 2 * W
        cy = (y0 + y1) / 2 * H
        pos = order_pos[elem_idx]

        r = 16
        draw.ellipse([cx - r, cy - r, cx + r, cy + r],
                     fill=(255, 210, 0, 230), outline=(0, 0, 0, 255), width=2)
        num_str = str(pos + 1)
        tw, th = draw.textbbox((0, 0), num_str, font=font_bold)[2:]
        draw.text((cx - tw / 2, cy - th / 2 - 1), num_str,
                  fill=(0, 0, 0, 255), font=font_bold)

    # 图例（放在图片右下角）
    legend_items = list(CLASS_COLORS.items())
    max_len = max(draw.textbbox((0, 0), k, font=font)[2] for k, _ in legend_items)
    lx = W - max_len - 68
    ly = 40
    pad = 5

    draw.rectangle([lx - pad, ly - pad, W - 8, ly + len(legend_items) * 20 + pad + 2],
                   fill=(255, 255, 255, 245), outline=(0, 0, 0, 80), width=1)
    for i, (cls_name, color) in enumerate(legend_items):
        draw.rectangle([lx, ly + i * 20, lx + 13, ly + i * 20 + 13],
                       fill=color + (200,), outline=(0, 0, 0, 100), width=1)
        draw.text((lx + 19, ly + i * 20 + 1), cls_name,
                  fill=(0, 0, 0, 255), font=font)

    # 标题栏（黑色背景条）
    draw.rectangle([0, 0, W, 32], fill=(20, 20, 20, 255))
    num_elems = len(bboxes)
    num_classes = len(set(categories))
    draw.text((10, 8),
              f"{num_elems} elements  |  {num_classes} types  |  {len(order)} in reading order",
              fill=(255, 255, 255, 255), font=font)

    # 转回 RGB 保存
    img.convert("RGB").save(output_path, "JPEG", quality=90)
    return output_path


def main():
    parser = argparse.ArgumentParser(
        description="RT-DETR + Pointer Network 文档布局检测和阅读顺序"
    )
    parser.add_argument("--detector", default="models/layout_yolov8n.pt",
                       help="检测模型路径（YOLO或RT-DETR）")
    parser.add_argument("--detector-type", default="auto",
                       choices=["auto", "yolo", "rtdetr"],
                       help="检测模型类型（auto会根据文件名自动判断）")
    parser.add_argument("--pointer", default=None,
                       help="指针网络模型路径（可选）")
    parser.add_argument("--pdf", required=True, help="PDF 文件")
    parser.add_argument("--page", type=int, default=0, help="页码（从0开始）")
    parser.add_argument("--dpi", type=int, default=150, help="渲染 DPI")
    parser.add_argument("--conf", type=float, default=0.3, help="检测置信度阈值")
    parser.add_argument("--output", "-o", default="output.jpg", help="输出图片")
    parser.add_argument("--dmodel", type=int, default=256, help="Pointer Network d_model")
    args = parser.parse_args()

    # ── 1. 加载模型 ──────────────────────────────────────────────────────────
    from ultralytics import RTDETR

    detector_type = args.detector_type
    if detector_type == "auto":
        if "rtdetr" in args.detector.lower():
            detector_type = "rtdetr"
        else:
            detector_type = "yolo"

    print(f"加载检测模型: {args.detector} (type: {detector_type})")
    if detector_type == "rtdetr":
        detector = RTDETR(args.detector)
    else:
        detector = YOLO(args.detector)

    pointer_model = None
    if args.pointer and Path(args.pointer).exists():
        print(f"加载指针网络: {args.pointer}")
        pointer_model = ReadingOrderModel(
            num_classes=NUM_CLASSES,
            d_model=args.dmodel,
            checkpoint_path=args.pointer,
        )
    else:
        print("使用启发式读序（Pointer Network 未加载）")

    # ── 2. 解析 PDF ───────────────────────────────────────────────────────────
    print(f"解析 PDF: {args.pdf}")
    with PDFParser(args.pdf) as parser:
        pages = parser.parse_all()
        if args.page >= len(pages):
            print(f"错误：PDF 只有 {len(pages)} 页")
            return

        page_info = pages[args.page]
        img_bytes = parser.render_page_to_image(args.page, dpi=args.dpi)

    img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    print(f"图片尺寸: {img.size}")

    # ── 3. 检测 ───────────────────────────────────────────────────────────────
    print("检测元素...")
    bboxes, cat_ids, confs = detect_elements(detector, img, conf_threshold=args.conf)
    print(f"检测到 {len(bboxes)} 个元素")

    # ── 4. 阅读顺序 ───────────────────────────────────────────────────────────
    print("预测阅读顺序...")
    order = predict_order(pointer_model, bboxes, cat_ids)
    print(f"阅读顺序: {order}")

    # ── 5. 可视化 ────────────────────────────────────────────────────────────
    print(f"保存结果: {args.output}")
    visualize(img, bboxes, cat_ids, confs, order, output_path=args.output)

    # ── 6. 打印结果 ───────────────────────────────────────────────────────────
    print("\n检测结果:")
    for i, (bbox, cat_id, conf) in enumerate(zip(bboxes, cat_ids, confs)):
        cls_name = CLASS_NAMES[cat_id] if cat_id < len(CLASS_NAMES) else "?"
        order_idx = order.index(i) if i in order else -1
        print(f"  [{i}] {cls_name:10s} conf={conf:.3f}  order=#{order_idx+1 if order_idx>=0 else '?'} "
              f"  bbox=[{bbox[0]:.2f},{bbox[1]:.2f},{bbox[2]:.2f},{bbox[3]:.2f}]")

    print(f"\n阅读顺序详情:")
    for pos, elem_idx in enumerate(order):
        cls_name = CLASS_NAMES[cat_ids[elem_idx]] if cat_ids[elem_idx] < len(CLASS_NAMES) else "?"
        conf = confs[elem_idx]
        print(f"  {pos+1}. {cls_name:10s} (elem #{elem_idx}, conf={conf:.3f})")


if __name__ == "__main__":
    main()
