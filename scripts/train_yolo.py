#!/usr/bin/env python3
"""
YOLO 布局检测模型训练
支持两种运行方式：
  1. 普通 CLI：  python scripts/train_yolo.py --epochs 100
  2. tmux 模式：python scripts/train_yolo.py --flagfile=/tmp/tmux_xxx_flags.json

flagfile 为 JSON 格式：
  {"data_dir": "...", "epochs": 100, "batch": 16, "model_size": "m"}
"""
import argparse
import os
import sys
import json
import shutil
from pathlib import Path


def get_project_root():
    return Path(__file__).parent.parent


def load_from_flagfile(path: str) -> dict:
    """从 flagfile 加载参数（tmux 模式）"""
    with open(path) as f:
        return json.load(f)


def check_dependencies():
    """检查依赖"""
    print("=== 检查依赖 ===")
    try:
        import ultralytics
        print(f"  ✓ ultralytics {ultralytics.__version__}")
    except ImportError:
        print("  ✗ ultralytics 未安装")
        print("  运行: python scripts/run_tmux.py install")
        sys.exit(1)

    try:
        import torch
        cuda = torch.cuda.is_available()
        print(f"  ✓ PyTorch {torch.__version__}" +
              (" (CUDA)" if cuda else " (CPU only)"))
    except ImportError:
        print("  ✗ PyTorch 未安装")
        sys.exit(1)


def prepare_dataset(data_dir: str) -> str:
    """准备/验证 YOLO 格式数据集，生成 data.yaml"""
    root = get_project_root()
    dataset_root = Path(data_dir).expanduser().resolve()

    if not dataset_root.exists():
        print(f"  ✗ 数据目录不存在: {dataset_root}")
        print(f"  请先准备好标注数据（使用 label_converter.py 转换）")
        sys.exit(1)

    # 类别定义（8类）
    classes = [
        "title", "header", "body",
        "figure", "caption", "table",
        "footnote", "equation",
    ]

    yaml_content = f"""
path: {dataset_root}
train: images/train
val: images/val

nc: {len(classes)}
names: {classes}
"""

    yaml_path = dataset_root / "data.yaml"
    with open(yaml_path, 'w') as f:
        f.write(yaml_content)

    print(f"  ✓ dataset root : {dataset_root}")
    print(f"  ✓ data.yaml   : {yaml_path}")
    return str(yaml_path)


def train(data_yaml: str, model_size: str = "m",
          epochs: int = 100, batch: int = 16,
          resume_path: str = None):
    """训练 YOLO 模型"""
    from ultralytics import YOLO

    root = get_project_root()
    run_dir = root / "runs" / "detect" / "layout_model"

    print(f"\n=== 开始训练 ===")
    print(f"  数据集   : {data_yaml}")
    print(f"  模型大小 : yolov8{model_size}")
    print(f"  epochs   : {epochs}")
    print(f"  batch    : {batch}")
    print(f"  输出目录  : {run_dir}")

    # 如果有已训练的模型，支持 resume
    if resume_path and Path(resume_path).exists():
        print(f"  恢复训练  : {resume_path}")
        model = YOLO(resume_path)
    else:
        # 从头训练（使用官方预训练权重做初始化）
        model = YOLO(f"yolov8{model_size}.pt")

    results = model.train(
        data=data_yaml,
        epochs=epochs,
        batch=batch,
        imgsz=640,
        patience=10,
        save=True,
        save_period=max(10, epochs // 10),
        project=str(run_dir.parent),
        name=run_dir.name,
        exist_ok=True,
        optimizer="AdamW",
        lr0=0.001,
        lrf=0.01,
        warmup_epochs=3,
        mosaic=0.5,
        mixup=0.1,
        close_mosaic=10,
        workers=4,
        device=0 if os.path.exists("/dev/nvidia0") else "cpu",
        verbose=True,
        seed=42,
    )

    # ── 保存最终模型 ─────────────────────────────────────────
    best_model = run_dir / "weights" / "best.pt"
    last_model = run_dir / "weights" / "last.pt"
    models_dir = root / "models"
    models_dir.mkdir(exist_ok=True)

    final_model = models_dir / f"layout_yolov8{model_size}.pt"
    if best_model.exists():
        shutil.copy(best_model, final_model)
        print(f"\n  ✓ 最佳模型: {final_model}")
    elif last_model.exists():
        shutil.copy(last_model, final_model)
        print(f"\n  ✓ 最终模型: {final_model}")

    # ── 打印指标 ─────────────────────────────────────────────
    if results and hasattr(results, 'box'):
        print(f"\n=== 训练完成 ===")
        print(f"  mAP50    : {results.box.map50:.4f}")
        print(f"  mAP50-95 : {results.box.map:.4f}")

    return results


def main():
    parser = argparse.ArgumentParser(description="训练文档布局检测 YOLO 模型")
    parser.add_argument("--flagfile", type=str,
                        help="tmux 模式：JSON 参数文件")
    parser.add_argument("--data-dir", type=str, default="./data/yolo_dataset")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--model-size", type=str, default="m",
                        choices=["n", "s", "m", "l", "x"])
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()

    # ── flagfile 模式（tmux 调用）────────────────────────────
    if args.flagfile:
        flags = load_from_flagfile(args.flagfile)
        data_dir = flags.get("data_dir", args.data_dir)
        epochs = flags.get("epochs", args.epochs)
        batch = flags.get("batch", args.batch)
        model_size = flags.get("model_size", args.model_size)
    else:
        data_dir = args.data_dir
        epochs = args.epochs
        batch = args.batch
        model_size = args.model_size

    check_dependencies()

    if args.check_only:
        print("依赖检查通过")
        return

    data_yaml = prepare_dataset(data_dir)
    train(data_yaml, model_size=model_size,
          epochs=epochs, batch=batch)


if __name__ == "__main__":
    main()
