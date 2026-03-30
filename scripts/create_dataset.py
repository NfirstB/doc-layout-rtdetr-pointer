#!/usr/bin/env python3
"""
create_dataset.py — 用 PyMuPDF 从已有 PDF 生成高质量布局数据集

用法：
  python scripts/create_dataset.py --input /path/to/pdfs --output ./dataset

输出：YOLO 格式 images/ + labels/
"""
import argparse
import io
import os
import sys
from pathlib import Path
from datetime import datetime
from collections import Counter, defaultdict

import fitz  # PyMuPDF
from PIL import Image


# ── 类别映射 ─────────────────────────────────────────────────────────────────
# 8 类（与 doc-layout-analyzer 训练目标一致）
CLASSES = ["title", "header", "body", "figure", "caption", "table", "footnote", "equation"]
CLASS_IDS = {name: i for i, name in enumerate(CLASSES)}

# PyMuPDF 字体 → 类别映射（启发式）
TITLE_FONTS = ['bold', 'black', 'title', 'heading', 'head', 'biolinum', 'helvetica-bold']
BODY_FONTS = ['regular', 'normal', 'libertine', 'times', 'roman', 'courier']


def get_font_class(font_name, size, max_size):
    """根据字体名和大小判断类别"""
    fn = font_name.lower() if font_name else ''
    
    # 特大字体 → title
    if size >= max_size * 0.9 and size > 14:
        return 'title'
    # 较大字体在顶部 → header
    if size >= max_size * 0.7 and size > 11:
        return 'header'
    # 小字体底部 → footnote
    if size < 9 and size > 0:
        return 'footnote'
    # 正文
    return 'body'


def extract_spans(page):
    """用 PyMuPDF 提取带字体信息的文本span"""
    text_dict = page.get_text("dict")
    spans = []
    for block in text_dict.get("blocks", []):
        if block.get("type") != 0:
            continue
        for line in block.get("lines", []):
            for span in line.get("spans", []):
                text = span.get('text', '').strip()
                if not text or len(text) < 2:
                    continue
                bbox = span.get('bbox', [])
                if len(bbox) != 4:
                    continue
                spans.append({
                    'text': text,
                    'size': span.get('size', 9),
                    'font': span.get('font', ''),
                    'color': span.get('color', 0),
                    'x0': bbox[0],
                    'y0': bbox[1],
                    'x1': bbox[2],
                    'y1': bbox[3],
                })
    return spans


def extract_images(page, page_height):
    """提取页面中的图片区域（PyMuPDF）"""
    img_list = page.get_images(full=True)
    images = []
    for img in img_list:
        xref = img[0]
        try:
            base = page.parent.extract_image(xref)
            img_w = base.get('width', 0)
            img_h = base.get('height', 0)
            # 图片在页面的位置通过 parent object 获取
            for block in page.get_text("dict")["blocks"]:
                if block.get("type") == 1:  # image block
                    bbox = block.get("bbox", [])
                    if bbox[2] - bbox[0] == img_w and bbox[3] - bbox[1] == img_h:
                        images.append({
                            'x0': bbox[0], 'y0': bbox[1],
                            'x1': bbox[2], 'y1': bbox[3],
                            'width': img_w, 'height': img_h
                        })
                        break
        except Exception:
            pass
    return images


def analyze_layout(page, dpi=150):
    """分析页面布局，返回 YOLO 格式标注"""
    W_pdf = float(page.rect.width)
    H_pdf = float(page.rect.height)

    spans = extract_spans(page)
    if not spans:
        return [], {}

    # 字体大小分析
    sizes = [s['size'] for s in spans if s['size'] > 0]
    max_size = max(sizes) if sizes else 12
    size_counter = Counter(round(s) for s in sizes)
    body_size = size_counter.most_common(1)[0][0] if size_counter else 9

    # 页面高度
    page_top = min(s['y0'] for s in spans)
    page_bottom = max(s['y1'] for s in spans)
    page_height = page_bottom - page_top

    # 按 y 坐标分行
    lines = defaultdict(list)
    for s in spans:
        # 归入行（y 坐标相近的归为一行）
        row_key = round(s['y0'] / 5) * 5
        lines[row_key].append(s)

    elements = []
    handled_spans = set()

    # ── 1. Title：最大字体在页面上部 ───────────────────────────────────
    title_spans = [s for s in spans
                   if s['size'] >= max_size * 0.85
                   and s['y0'] < page_height * 0.15
                   and len(s['text']) > 3]
    if title_spans:
        min_y = min(s['y0'] for s in title_spans)
        max_y = max(s['y1'] for s in title_spans)
        min_x = min(s['x0'] for s in title_spans)
        max_x = max(s['x1'] for s in title_spans)
        # 扩展边界
        elements.append({
            'class': 'title',
            'x0': max(0, min_x - 2),
            'y0': max(0, min_y - 2),
            'x1': min(W_pdf, max_x + 2),
            'y1': min(H_pdf, max_y + 2),
        })
        for s in title_spans:
            handled_spans.add(id(s))

    # ── 2. Header：顶部中等偏大字体的单行 ─────────────────────────────
    header_spans = [s for s in spans
                    if s['size'] >= body_size * 1.1
                    and s['y0'] < page_height * 0.2
                    and s['y0'] >= page_height * 0.05
                    and id(s) not in handled_spans
                    and len(s['text']) > 3]
    if header_spans:
        # 按 x 分组（可能有两栏）
        header_spans_sorted = sorted(header_spans, key=lambda s: s['x0'])
        mid_x = W_pdf / 2
        left = [s for s in header_spans_sorted if s['x1'] < mid_x]
        right = [s for s in header_spans_sorted if s['x0'] > mid_x]
        for group in [left, right]:
            if not group:
                continue
            min_y = min(s['y0'] for s in group)
            max_y = max(s['y1'] for s in group)
            min_x = min(s['x0'] for s in group)
            max_x = max(s['x1'] for s in group)
            elements.append({
                'class': 'header',
                'x0': max(0, min_x - 2),
                'y0': max(0, min_y - 1),
                'x1': min(W_pdf, max_x + 2),
                'y1': min(H_pdf, max_y + 1),
            })
            for s in group:
                handled_spans.add(id(s))

    # ── 3. Body：主体文字块 ────────────────────────────────────────────
    body_spans = [s for s in spans
                  if id(s) not in handled_spans
                  and s['size'] >= body_size * 0.7
                  and s['y0'] > page_height * 0.18
                  and s['y1'] < page_height * 0.90
                  and len(s['text']) > 5]

    # 聚合成行
    line_rows = defaultdict(list)
    for s in body_spans:
        row_key = round(s['y0'] / 6) * 6
        line_rows[row_key].append(s)

    # 聚合成列（两栏布局）
    for row_key, row_spans in sorted(line_rows.items()):
        if not row_spans:
            continue
        row_spans_sorted = sorted(row_spans, key=lambda s: s['x0'])
        mid_x = W_pdf * 0.45  # 两栏分割线

        for col_spans in [row_spans_sorted[:2], row_spans_sorted[2:4]]:
            if not col_spans:
                continue
            # 找这一行属于哪一列
            col_min_x = min(s['x0'] for s in row_spans_sorted)
            col_max_x = max(s['x1'] for s in row_spans_sorted)
            is_left_col = col_min_x < mid_x

            if is_left_col:
                col_spans_in = [s for s in row_spans if s['x1'] < mid_x]
            else:
                col_spans_in = [s for s in row_spans if s['x0'] > mid_x * 0.8]

            if not col_spans_in:
                continue

            min_y = min(s['y0'] for s in col_spans_in)
            max_y = max(s['y1'] for s in col_spans_in)
            min_x = min(s['x0'] for s in col_spans_in)
            max_x = max(s['x1'] for s in col_spans_in)

            if max_y - min_y > 3 and max_x - min_x > 10:
                elements.append({
                    'class': 'body',
                    'x0': max(0, min_x - 1),
                    'y0': max(0, min_y - 1),
                    'x1': min(W_pdf, max_x + 1),
                    'y1': min(H_pdf, max_y + 1),
                })
                for s in col_spans_in:
                    handled_spans.add(id(s))

    # ── 4. Footnote：底部小字体 ───────────────────────────────────────
    footnote_spans = [s for s in spans
                     if s['size'] < body_size * 0.85
                     and s['y0'] > page_height * 0.88
                     and id(s) not in handled_spans
                     and len(s['text']) > 3]
    if footnote_spans:
        min_y = min(s['y0'] for s in footnote_spans)
        max_y = max(s['y1'] for s in footnote_spans)
        min_x = min(s['x0'] for s in footnote_spans)
        max_x = max(s['x1'] for s in footnote_spans)
        elements.append({
            'class': 'footnote',
            'x0': max(0, min_x - 2),
            'y0': max(0, min_y - 1),
            'x1': min(W_pdf, max_x + 2),
            'y1': min(H_pdf, max_y + 1),
        })
        for s in footnote_spans:
            handled_spans.add(id(s))

    # ── 5. Figure/Table：从图片块获取 ─────────────────────────────────
    images = extract_images(page, page_height)
    for img in images:
        img_w = img['x1'] - img['x0']
        img_h = img['y1'] - img['y0']
        if img_w < W_pdf * 0.05 or img_h < H_pdf * 0.02:
            continue
        # 根据宽高比判断
        aspect = img_w / (img_h + 1e-6)
        cls = 'table' if (aspect < 0.7 and img_h > H_pdf * 0.08) else 'figure'
        elements.append({
            'class': cls,
            'x0': max(0, img['x0']),
            'y0': max(0, img['y0']),
            'x1': min(W_pdf, img['x1']),
            'y1': min(H_pdf, img['y1']),
        })

    # ── 6. 过滤无效 ────────────────────────────────────────────────────
    elements = [e for e in elements
                 if e['x1'] > e['x0'] + 5 and e['y1'] > e['y0'] + 3]

    # 坐标归一化到 [0, 1]
    for e in elements:
        e['x0'] = max(0, min(1, e['x0'] / W_pdf))
        e['y0'] = max(0, min(1, e['y0'] / H_pdf))
        e['x1'] = max(0, min(1, e['x1'] / W_pdf))
        e['y1'] = max(0, min(1, e['y1'] / H_pdf))

    return elements, {'max_size': max_size, 'body_size': body_size}


def render_page(page, dpi=150):
    """将页面渲染为 PIL Image"""
    mat = fitz.Matrix(dpi/72, dpi/72)
    pix = page.get_pixmap(matrix=mat, alpha=False)
    img_bytes = pix.tobytes("jpg")
    return Image.open(io.BytesIO(img_bytes))


# ── 批量处理 ─────────────────────────────────────────────────────────────────

def process_pdfs(pdf_paths, output_dir, dpi=150, max_pages=None, sample_rate=1):
    """处理多个 PDF"""
    out_img_train = Path(output_dir) / "images" / "train"
    out_lbl_train = Path(output_dir) / "labels" / "train"
    out_img_val = Path(output_dir) / "images" / "val"
    out_lbl_val = Path(output_dir) / "labels" / "val"

    for d in [out_img_train, out_lbl_train, out_img_val, out_lbl_val]:
        d.mkdir(parents=True, exist_ok=True)

    total_pages = 0
    total_elements = []

    for pdf_path in pdf_paths:
        stem = Path(pdf_path).stem
        try:
            doc = fitz.open(pdf_path)
        except Exception as e:
            print(f"  ✗ 打开失败: {e}")
            continue

        num_pages = len(doc)
        pages_to_process = list(range(min(num_pages, max_pages or num_pages)))
        if sample_rate < 1:
            pages_to_process = [p for i, p in enumerate(pages_to_process)
                               if i % int(1/sample_rate) == 0]

        is_val = total_pages > 50  # 前50页用于val

        for page_num in pages_to_process:
            page = doc[page_num]

            # 渲染图片
            img = render_page(page, dpi)
            suffix = "val" if is_val else "train"
            img_path = Path(output_dir) / "images" / suffix / f"{stem}_p{page_num+1:03d}.jpg"
            img.save(img_path, "JPEG", quality=88)

            # 分析布局
            elements, font_info = analyze_layout(page, dpi)
            total_elements.extend(elements)

            # 写入标签
            lbl_path = Path(output_dir) / "labels" / suffix / f"{stem}_p{page_num+1:03d}.txt"
            with open(lbl_path, 'w') as f:
                for elem in elements:
                    cls_id = CLASS_IDS.get(elem['class'], -1)
                    if cls_id < 0:
                        continue
                    cx = (elem['x0'] + elem['x1']) / 2
                    cy = (elem['y0'] + elem['y1']) / 2
                    bw = elem['x1'] - elem['x0']
                    bh = elem['y1'] - elem['y0']
                    if bw > 0.005 and bh > 0.005:
                        f.write(f"{cls_id} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n")

            total_pages += 1
            is_val = total_pages > 50

        doc.close()

    return total_pages, total_elements


def create_yaml(output_dir):
    yaml_content = f"""path: {Path(output_dir).resolve()}
train: images/train
val: images/val

nc: {len(CLASSES)}
names: {CLASSES}
"""
    yaml_path = Path(output_dir) / 'data.yaml'
    with open(yaml_path, 'w') as f:
        f.write(yaml_content)
    return yaml_path


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="从 PDF 生成 YOLO 布局数据集")
    parser.add_argument("--input", "-i", required=True, help="PDF 文件或目录")
    parser.add_argument("--output", "-o", required=True, help="输出目录")
    parser.add_argument("--dpi", type=int, default=150, help="渲染 DPI")
    parser.add_argument("--max-pages", type=int, default=None, help="每个 PDF 最大页数")
    parser.add_argument("--sample-rate", type=float, default=0.3, help="采样率（默认0.3=每隔3页取1页）")
    args = parser.parse_args()

    input_path = Path(args.input)
    if input_path.is_file():
        pdfs = [input_path]
    else:
        pdfs = sorted(input_path.glob("*.pdf"))

    if not pdfs:
        print(f"✗ 未找到 PDF: {args.input}")
        return

    print(f"\n{'='*50}")
    print(f"  输入: {args.input}")
    print(f"  输出: {args.output}")
    print(f"  PDF数: {len(pdfs)}")
    print(f"  DPI:   {args.dpi}")
    print(f"  采样:  每{1/args.sample_rate:.0f}页取1页")
    print(f"{'='*50}\n")

    t0 = datetime.now()
    pages, elements = process_pdfs(
        [str(p) for p in pdfs],
        args.output,
        dpi=args.dpi,
        max_pages=args.max_pages,
        sample_rate=args.sample_rate,
    )

    elapsed = (datetime.now() - t0).total_seconds()

    # 统计
    class_counts = defaultdict(int)
    for e in elements:
        class_counts[e['class']] += 1

    print(f"\n{'='*50}")
    print(f"  ✓ 完成！")
    print(f"  总页数: {pages}")
    print(f"  总耗时: {elapsed:.1f}s")
    print(f"  类别统计:")
    for cls in CLASSES:
        print(f"    {cls:10s}: {class_counts.get(cls, 0):5d}")
    print(f"{'='*50}")

    yaml_path = create_yaml(args.output)
    print(f"\n✓ data.yaml: {yaml_path}")


if __name__ == "__main__":
    main()
