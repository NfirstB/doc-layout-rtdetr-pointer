"""
Label-Studio 标注格式转换器

支持:
1. Label-Studio JSON 导出 → YOLO 格式
2. YOLO 格式 → Label-Studio JSON 导入格式
3. 生成 Label-Studio 配置 XML

使用方法:
  python scripts/label_converter.py --input ./ls_export.json --output ./yolo_dataset/
  python scripts/label_converter.py --to-ls --input ./yolo_dataset/ --output ./ls_import.json
"""
import argparse
import json
import os
import sys
import shutil
from pathlib import Path
import xml.etree.ElementTree as ET
from typing import List, Dict, Tuple


# Label-Studio 项目配置模板（8类）
LS_CONFIG_TEMPLATE = """<?xml version="1.0" encoding="UTF-8"?>
<View>
  <Header value="文档布局标注"/>
  <Image name="image" value="$image"/>
  <Label>
    <Label value="title" background="#FF6B6B"/>
    <Label value="header" background="#4ECDC4"/>
    <Label value="body" background="#45B7D1"/>
    <Label value="figure" background="#96CEB4"/>
    <Label value="caption" background="#FFEAA7"/>
    <Label value="table" background="#DDA0DD"/>
    <Label value="footnote" background="#98D8C8"/>
    <Label value="equation" background="#F7DC6F"/>
  </Label>
  <RectangleLabels name="label" toName="image">
    <Label value="title" background="#FF6B6B"/>
    <Label value="header" background="#4ECDC4"/>
    <Label value="body" background="#45B7D1"/>
    <Label value="figure" background="#96CEB4"/>
    <Label value="caption" background="#FFEAA7"/>
    <Label value="table" background="#DDA0DD"/>
    <Label value="footnote" background="#98D8C8"/>
    <Label value="equation" background="#F7DC6F"/>
  </RectangleLabels>
</View>
"""


class LabelStudioConverter:
    """Label-Studio ↔ YOLO 格式互转"""

    CLASSES = ["title", "header", "body", "figure", "caption", "table", "footnote", "equation"]

    def __init__(self, img_width: int, img_height: int):
        self.img_width = img_width
        self.img_height = img_height

    @staticmethod
    def ls_json_to_yolo(ls_json: Dict, output_dir: str) -> bool:
        """
        将 Label-Studio JSON 导出转换为 YOLO TXT 格式
        ls_json: Label-Studio 导出的单个任务 JSON
        """
        # 创建输出目录
        os.makedirs(f"{output_dir}/images", exist_ok=True)
        os.makedirs(f"{output_dir}/labels", exist_ok=True)

        tasks = ls_json if isinstance(ls_json, list) else [ls_json]
        count = 0

        for task in tasks:
            # 获取图片路径
            if "data" in task:
                data = task["data"]
                img_path = data.get("image", "")
                if img_path.startswith("http"):
                    img_path = Path(img_path).name
            elif "image" in task:
                img_path = task["image"]
            else:
                continue

            # 解析标注
            if "annotations" not in task or not task["annotations"]:
                continue

            for ann in task["annotations"]:
                if "result" not in ann:
                    continue

                stem = Path(img_path).stem
                label_path = f"{output_dir}/labels/{stem}.txt"

                with open(label_path, 'w') as f:
                    for result in ann["result"]:
                        if result.get("type") != "rectanglelabels":
                            continue

                        # 坐标是百分比形式（Label-Studio 默认）
                        x = result["value"]["x"] / 100  # 0~1
                        y = result["value"]["y"] / 100
                        w = result["value"]["width"] / 100
                        h = result["value"]["height"] / 100

                        # YOLO 格式：class_id cx cy w h（全是归一化值）
                        cx = x + w / 2
                        cy = y + h / 2

                        labels = result["value"].get("rectanglelabels", [])
                        if not labels:
                            continue

                        label = labels[0]
                        if label not in LabelStudioConverter.CLASSES:
                            continue

                        cls_id = LabelStudioConverter.CLASSES.index(label)
                        f.write(f"{cls_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n")

                # 复制图片
                src_img = img_path
                if os.path.exists(src_img):
                    shutil.copy(src_img, f"{output_dir}/images/{Path(img_path).name}")
                    count += 1

        print(f"  ✓ 转换完成: {count} 张图片")
        return count > 0

    @staticmethod
    def yolo_to_ls_json(yolo_dir: str, images_dir: str) -> List[Dict]:
        """将 YOLO 格式转换为 Label-Studio 可导入的 JSON"""
        results = []
        label_files = sorted(Path(yolo_dir).glob("*.txt"))

        for lf in label_files:
            stem = lf.stem
            img_name = f"{stem}.jpg"
            img_path = Path(images_dir) / img_name

            # 读取标注
            yolo_lines = []
            if lf.exists():
                with open(lf) as f:
                    yolo_lines = f.readlines()

            # 构建 LS result 格式
            ls_results = []
            for line in yolo_lines:
                parts = line.strip().split()
                if len(parts) != 5:
                    continue
                cls_id, cx, cy, w, h = map(float, parts)
                cls_id = int(cls_id)

                if cls_id >= len(LabelStudioConverter.CLASSES):
                    continue

                # 转换为百分比
                x_pct = (cx - w/2) * 100
                y_pct = (cy - h/2) * 100
                w_pct = w * 100
                h_pct = h * 100

                ls_results.append({
                    "type": "rectanglelabels",
                    "from_name": "label",
                    "to_name": "image",
                    "value": {
                        "x": x_pct,
                        "y": y_pct,
                        "width": w_pct,
                        "height": h_pct,
                        "rotation": 0,
                        "rectanglelabels": [LabelStudioConverter.CLASSES[cls_id]],
                    }
                })

            if not ls_results:
                continue

            task = {
                "data": {"image": str(img_path)},
                "annotations": [{
                    "result": ls_results
                }]
            }
            results.append(task)

        return results


def generate_ls_config(output_path: str = "label_config.xml"):
    """生成 Label-Studio 项目配置文件"""
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(LS_CONFIG_TEMPLATE)
    print(f"  ✓ Label-Studio 配置已生成: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="标注格式转换")
    parser.add_argument("--input", type=str, required=True, help="输入文件/目录")
    parser.add_argument("--output", type=str, required=True, help="输出目录")
    parser.add_argument("--to-ls", action="store_true", help="YOLO → Label-Studio 格式")
    parser.add_argument("--to-yolo", action="store_true", help="Label-Studio → YOLO 格式")
    parser.add_argument("--gen-config", action="store_true", help="生成 Label-Studio 配置")
    parser.add_argument("--img-width", type=int, default=595, help="图片宽度（磅，PDF pt）")
    parser.add_argument("--img-height", type=int, default=842, help="图片高度")
    args = parser.parse_args()

    if args.gen_config:
        generate_ls_config(args.output)
        return

    if args.to_yolo:
        with open(args.input) as f:
            ls_data = json.load(f)
        converter = LabelStudioConverter(args.img_width, args.img_height)
        converter.ls_json_to_yolo(ls_data, args.output)

    elif args.to_ls:
        images_dir = Path(args.input).parent / "images"
        results = LabelStudioConverter.yolo_to_ls_json(args.input, str(images_dir))
        with open(args.output, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"  ✓ 已导出 {len(results)} 条任务到 {args.output}")

    else:
        print("请指定 --to-yolo 或 --to-ls")


if __name__ == "__main__":
    main()
