#!/usr/bin/env python3
"""
生成合成文档布局数据集（Demo 用）

用 Pillow 生成带布局的文档图片 + YOLO 标注，
每张图片包含：标题、正文段落、图片块、表格块。

用法：
  python scripts/download_publaynet.py
  python scripts/download_publaynet.py --max-images 300 --output ./data/my_dataset/
"""
import argparse
import random
import shutil
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


OUR_CLASSES = ["title", "header", "body", "figure", "caption", "table", "footnote", "equation"]
CLASS_IDS = {c: i for i, c in enumerate(OUR_CLASSES)}


def create_synthetic_demo(output_dir: Path, n_images: int = 300):
    """
    生成 n_images 张合成文档图片
    输出 YOLO 格式：
      data/yolo_dataset/images/train/*.jpg
      data/yolo_dataset/labels/train/*.txt
      data/yolo_dataset/images/val/*.jpg
      data/yolo_dataset/labels/val/*.txt
    """
    W, H = 595, 842  # A4 @ 72dpi

    img_train_dir = output_dir / "images" / "train"
    img_val_dir   = output_dir / "images" / "val"
    lbl_train_dir = output_dir / "labels" / "train"
    lbl_val_dir   = output_dir / "labels" / "val"

    for d in [img_train_dir, img_val_dir, lbl_train_dir, lbl_val_dir]:
        d.mkdir(parents=True, exist_ok=True)

    n_train = int(n_images * 0.8)
    print(f"  生成 {n_train} 张训练图片...")
    for i in range(n_train):
        img, lines = _make_one_page(seed=i * 1000 + 42, page_num=i)
        img.save(img_train_dir / f"train_{i:04d}.jpg", "JPEG", quality=85)
        with open(lbl_train_dir / f"train_{i:04d}.txt", "w") as f:
            f.write("\n".join(lines) + "\n")
        if (i + 1) % 50 == 0:
            print(f"    {i+1}/{n_train}")

    print(f"  生成 {n_images - n_train} 张验证图片...")
    for i in range(n_images - n_train):
        img, lines = _make_one_page(seed=i * 7777 + 13, page_num=n_train + i)
        img.save(img_val_dir / f"val_{i:04d}.jpg", "JPEG", quality=85)
        with open(lbl_val_dir / f"val_{i:04d}.txt", "w") as f:
            f.write("\n".join(lines) + "\n")

    # 生成 data.yaml
    yaml_content = f"""# Auto-generated
path: {output_dir.resolve()}
train: images/train
val: images/val

nc: {len(OUR_CLASSES)}
names: {OUR_CLASSES}
"""
    with open(output_dir / "data.yaml", "w") as f:
        f.write(yaml_content)

    print(f"  ✓ 训练集 {n_train} 张，验证集 {n_images - n_train} 张")
    print(f"  ✓ 类别: {OUR_CLASSES}")


def _make_one_page(seed: int, page_num: int):
    """生成一张合成文档页面"""
    random.seed(seed)
    W, H = 595, 842
    img = Image.new("RGB", (W, H), "white")
    draw = ImageDraw.Draw(img)
    yolo_lines = []
    y = 20

    # ── 标题（title, id=0）────────────────────────────────────────────
    title_h = random.randint(25, 40)
    title_w = random.randint(300, 520)
    title_x = random.randint(30, W - title_w - 30)
    draw.rectangle([title_x, y, title_x + title_w, y + title_h],
                   fill="#E8E8E8", outline="#999")
    draw.text((title_x + 8, y + 6), f"Section {page_num}: Title Text",
              fill="#222")
    _add_yolo(yolo_lines, 0, title_x, y, title_w, title_h, W, H)
    y += title_h + 18

    # ── 章节标题（header, id=1）────────────────────────────────────────
    header_h = 22
    header_w = random.randint(200, 400)
    header_x = random.randint(30, W - header_w - 30)
    draw.rectangle([header_x, y, header_x + header_w, y + header_h],
                   fill="#D0D8E8", outline="#88A")
    draw.text((header_x + 6, y + 4), f"{page_num}.1 Introduction",
              fill="#111")
    _add_yolo(yolo_lines, 1, header_x, y, header_w, header_h, W, H)
    y += header_h + 12

    # ── 正文段落（body, id=2）───────────────────────────────────────────
    n_paras = random.randint(3, 5)
    for _ in range(n_paras):
        para_h = random.randint(35, 70)
        if y + para_h > H - 80:
            break
        draw.rectangle([30, y, W - 30, y + para_h],
                       fill="white", outline="#CCC")
        # 随机文字
        words = " ".join(["example" for _ in range(random.randint(20, 40))])
        draw.text((35, y + 5), words[:80], fill="#333")
        _add_yolo(yolo_lines, 2, 30, y, W - 60, para_h, W, H)
        y += para_h + 8

    # ── 图片块（figure, id=3）────────────────────────────────────────
    fig_w = random.randint(180, 350)
    fig_h = random.randint(100, 180)
    fig_x = random.randint(30, W - fig_w - 30)
    if y + fig_h < H - 60:
        # 图片占位（随机色块）
        for row in range(0, fig_h, 20):
            for col in range(0, fig_w, 20):
                c = tuple(random.randint(180, 240) for _ in range(3))
                draw.rectangle([fig_x + col, y + row,
                                 fig_x + min(col+18, fig_w),
                                 y + min(row+18, fig_h)], fill=c)
        draw.rectangle([fig_x, y, fig_x + fig_w, y + fig_h],
                       outline="#3366AA", width=2)
        _add_yolo(yolo_lines, 3, fig_x, y, fig_w, fig_h, W, H)

        # 图注（caption, id=4）
        cap_h = 18
        draw.rectangle([fig_x, y + fig_h, fig_x + fig_w, y + fig_h + cap_h],
                       fill="#FFF8DC", outline="#DAA520")
        draw.text((fig_x + 5, y + fig_h + 3),
                  f"Fig. {page_num}. Architecture diagram", fill="#555")
        _add_yolo(yolo_lines, 4, fig_x, y + fig_h, fig_w, cap_h, W, H)
        y += fig_h + cap_h + 15

    # ── 表格（table, id=5）────────────────────────────────────────────
    tbl_w = random.randint(350, 520)
    tbl_h = random.randint(80, 150)
    tbl_x = random.randint(30, W - tbl_w - 30)
    if y + tbl_h < H - 60:
        draw.rectangle([tbl_x, y, tbl_x + tbl_w, y + tbl_h],
                       fill="#F0FFFF", outline="#2E8B57", width=2)
        # 表格内部线
        n_rows = 3
        n_cols = 4
        row_h = tbl_h / n_rows
        col_w = tbl_w / n_cols
        for r in range(1, n_rows):
            draw.line([(tbl_x, y + r * row_h),
                       (tbl_x + tbl_w, y + r * row_h)],
                      fill="#2E8B57", width=1)
        for c in range(1, n_cols):
            draw.line([(tbl_x + c * col_w, y),
                       (tbl_x + c * col_w, y + tbl_h)],
                      fill="#2E8B57", width=1)
        # 填表头
        draw.rectangle([tbl_x, y, tbl_x + tbl_w, y + row_h],
                       fill="#98FB98", outline="#2E8B57")
        draw.text((tbl_x + 5, y + 4), f"Col1    Col2    Col3    Col4", fill="#000")
        _add_yolo(yolo_lines, 5, tbl_x, y, tbl_w, tbl_h, W, H)
        y += tbl_h + 15

    # ── 脚注（footnote, id=6）────────────────────────────────────────
    if y < H - 60:
        note_h = 20
        draw.rectangle([30, H - note_h - 10, W - 30, H - 10],
                       fill="#F5F5F5", outline="#AAA")
        draw.text((35, H - note_h - 6),
                  f"Reference: Paper {page_num} · Page {page_num} of demo dataset",
                  fill="#666")
        _add_yolo(yolo_lines, 6, 30, H - note_h - 10, W - 60, note_h, W, H)

    return img, yolo_lines


def _add_yolo(lines, cls_id, x, y_pos, w, h, img_w, img_h):
    """将边界框转为 YOLO 格式并添加到 lines"""
    cx = (x + w / 2) / img_w
    cy = (y_pos + h / 2) / img_h
    nw = w / img_w
    nh = h / img_h
    cx = min(1.0, max(0.0, cx))
    cy = min(1.0, max(0.0, cy))
    nw = min(1.0, max(0.0, nw))
    nh = min(1.0, max(0.0, nh))
    lines.append(f"{cls_id} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")


def main():
    parser = argparse.ArgumentParser(
        description="生成合成文档布局 Demo 数据集（YOLO 格式）")
    parser.add_argument("--output", type=str,
                        default="data/yolo_dataset",
                        help="输出目录（默认 data/yolo_dataset）")
    parser.add_argument("--max-images", type=int, default=300,
                        help="图片总数（默认 300）")
    args = parser.parse_args()

    root = Path(__file__).parent.parent
    output_dir = (root / args.output).expanduser().resolve()

    print("=" * 50)
    print("  合成文档布局 Demo 数据集生成器")
    print("=" * 50)
    print(f"  输出目录 : {output_dir}")
    print(f"  图片数量 : {args.max_images}")
    print()

    try:
        create_synthetic_demo(output_dir, n_images=args.max_images)
        print()
        print("=" * 50)
        print("  ✓ 数据集生成完成！")
        print("=" * 50)
        print(f"\n现在可以启动训练：")
        print(f"  python scripts/run_tmux.py train --epochs 50 --batch 16")
    except Exception as e:
        print(f"\n✗ 失败: {e}")
        import traceback; traceback.print_exc()


if __name__ == "__main__":
    main()
