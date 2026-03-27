#!/usr/bin/env python3
"""
YOLO 布局检测模型训练
使用方法:
  python scripts/train_yolo.py --data-dir ./data/annotated --epochs 100
"""
import argparse
import os
import sys
import subprocess
from pathlib import Path


def get_project_root():
    return Path(__file__).parent.parent


def check_dependencies():
    """检查依赖"""
    print("=== 检查依赖 ===")
    try:
        import ultralytics
        print(f"  ✓ ultralytics {ultralytics.__version__}")
    except ImportError:
        print("  ✗ ultralytics 未安装，正在安装...")
        subprocess.run([sys.executable, "-m", "pip", "install", "ultralytics", "-q"], check=True)

    try:
        import torch
        print(f"  ✓ PyTorch {torch.__version__} (CUDA: {torch.cuda.is_available()})")
    except ImportError:
        print("  ✗ PyTorch 未安装")
        sys.exit(1)


def prepare_dataset(data_yaml: str, images_dir: str, labels_dir: str) -> str:
    """准备 YOLO 格式数据集，生成 data.yaml"""
    root = get_project_root()
    dataset_root = root / "data" / "yolo_dataset"

    # 类别定义（8类）
    classes = [
        "title",       # 0 标题
        "header",      # 1 章节标题
        "body",        # 2 正文
        "figure",      # 3 图片
        "caption",     # 4 图片/表格标题
        "table",       # 5 表格
        "footnote",    # 6 脚注
        "equation",    # 7 公式
    ]

    yaml_content = f"""
# 文档布局检测数据集
path: {dataset_root}
train: images/train
val: images/val

nc: {len(classes)}
names: {classes}
"""

    yaml_path = dataset_root / "data.yaml"
    yaml_path.parent.mkdir(parents=True, exist_ok=True)
    with open(yaml_path, 'w') as f:
        f.write(yaml_content)

    print(f"  ✓ 数据集配置: {yaml_path}")
    return str(yaml_path)


def train(data_yaml: str, model_size: str = "m", epochs: int = 100, batch: int = 16):
    """训练 YOLO 模型"""
    from ultralytics import YOLO

    print(f"\n=== 开始训练 YOLO (epochs={epochs}, batch={batch}, size={model_size}) ===")

    model = YOLO(f"yolov8{model_size}.yaml")

    results = model.train(
        data=data_yaml,
        epochs=epochs,
        batch=batch,
        imgsz=640,
        patience=10,           # 早停
        save=True,
        save_period=10,        # 每10个epoch保存一次
        project=str(get_project_root() / "runs" / "detect"),
        name="layout_model",
        exist_ok=True,
        pretrained=False,
        optimizer="AdamW",
        lr0=0.001,
        lrf=0.01,
        warmup_epochs=3,
        mosaic=0.5,
        mixup=0.1,
        close_mosaic=10,
        Workers=4,
        device=0 if os.path.exists("/dev/nvidia0") else "cpu",
        verbose=True,
    )

    # 保存最终模型
    best_model = get_project_root() / "runs" / "detect" / "layout_model" / "weights" / "best.pt"
    final_model = get_project_root() / "models" / f"layout_yolov8{model_size}.pt"
    final_model.parent.mkdir(parents=True, exist_ok=True)
    if best_model.exists():
        import shutil
        shutil.copy(best_model, final_model)
        print(f"\n  ✓ 模型已保存: {final_model}")

    # 打印 mAP
    if results:
        print(f"\n=== 训练完成 ===")
        print(f"  mAP50: {results.box.map50:.4f}")
        print(f"  mAP50-95: {results.box.map:.4f}")

    return results


def main():
    parser = argparse.ArgumentParser(description="训练文档布局检测 YOLO 模型")
    parser.add_argument("--data-dir", type=str, default="./data/annotated",
                        help="标注数据目录（包含 images/ 和 labels/）")
    parser.add_argument("--epochs", type=int, default=100,
                        help="训练轮数")
    parser.add_argument("--batch", type=int, default=16,
                        help="批次大小")
    parser.add_argument("--model-size", type=str, default="m",
                        choices=["n", "s", "m", "l", "x"],
                        help="YOLO 模型大小")
    parser.add_argument("--check-only", action="store_true",
                        help="仅检查依赖，不训练")
    args = parser.parse_args()

    check_dependencies()

    if args.check_only:
        print("依赖检查通过")
        return

    data_yaml = prepare_dataset(args.data_dir, "", "")
    train(data_yaml, model_size=args.model_size, epochs=args.epochs, batch=args.batch)


if __name__ == "__main__":
    main()
