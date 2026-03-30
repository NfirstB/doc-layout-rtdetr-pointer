#!/usr/bin/env python3
"""
RT-DETR 训练脚本

用法：
  python scripts/train_rtdetr.py --data-dir /path/to/dataset --epochs 50 --model-size l
"""
import argparse
import os
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import torch
from ultralytics import RTDETR


# ── 依赖检查 ──────────────────────────────────────────────────────────────────

def check_deps():
    errors = []
    try:
        import ultralytics
        print(f"  ✓ ultralytics {ultralytics.__version__}")
    except ImportError:
        errors.append("  ✗ ultralytics 未安装")
    try:
        import torch
        cuda = torch.cuda.is_available()
        print(f"  ✓ PyTorch {torch.__version__}" + (" (CUDA)" if cuda else " (CPU)"))
    except ImportError:
        errors.append("  ✗ PyTorch 未安装")
    if errors:
        print("\n依赖检查失败：")
        for e in errors:
            print(e)
        sys.exit(1)


# ── 训练 ─────────────────────────────────────────────────────────────────────

def train(data_yaml: str, model_size: str = "l",
          epochs: int = 50, batch: int = 16,
          resume_path: str = None):
    """
    启动 RT-DETR 训练

    RT-DETR 是基于 Transformer 的实时检测器，
    比 YOLO 更适合复杂场景，但训练更慢。
    """
    print(f"\n{'='*50}")
    print(f"  RT-DETR 训练")
    print(f"  模型: rtdetr-{model_size}")
    print(f"  数据集: {data_yaml}")
    print(f"  epochs: {epochs}")
    print(f"  batch: {batch}")
    print(f"{'='*50}\n")

    # 加载模型
    if resume_path and Path(resume_path).exists():
        print(f"从检查点恢复: {resume_path}")
        model = RTDETR(resume_path)
    else:
        print(f"加载预训练 rtdetr-{model_size}...")
        model = RTDETR(f"rtdetr-{model_size}.pt")

    # 训练
    results = model.train(
        data=data_yaml,
        epochs=epochs,
        batch=batch,
        imgsz=640,
        patience=10,
        save=True,
        save_period=max(5, epochs // 10),
        project=str(Path(__file__).parent.parent / "runs" / "detect"),
        name="rtdetr_model",
        exist_ok=True,
        optimizer="AdamW",
        lr0=0.0001,
        lrf=0.01,
        warmup_epochs=3,
        workers=0,
        amp=True,
        device="0" if torch.cuda.is_available() else "cpu",
        verbose=True,
        seed=42,
    )

    # 保存最终模型
    best_pt = Path(__file__).parent.parent / "runs" / "detect" / "rtdetr_model" / "weights" / "best.pt"
    last_pt = Path(__file__).parent.parent / "runs" / "detect" / "rtdetr_model" / "weights" / "last.pt"
    models_dir = Path(__file__).parent.parent / "models"
    models_dir.mkdir(exist_ok=True)
    out_model = models_dir / f"rtdetr_{model_size}.pt"

    if best_pt.exists():
        import shutil
        shutil.copy(best_pt, out_model)
        print(f"\n✓ 最佳模型已保存: {out_model}")
    elif last_pt.exists():
        import shutil
        shutil.copy(last_pt, out_model)
        print(f"\n✓ 最终模型已保存: {out_model}")

    return out_model


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="训练 RT-DETR 文档布局检测模型")
    parser.add_argument("--data-dir", type=str, required=True,
                       help="YOLO 数据集根目录（含 data.yaml）")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--model-size", type=str, default="l",
                       choices=["l", "x"],
                       help="RT-DETR 模型大小")
    parser.add_argument("--resume", type=str, default=None,
                       help="从检查点恢复训练")

    args = parser.parse_args()

    check_deps()

    data_yaml = Path(args.data_dir) / "data.yaml"
    if not data_yaml.exists():
        print(f"✗ data.yaml 不存在: {data_yaml}")
        sys.exit(1)

    out = train(
        data_yaml=str(data_yaml),
        model_size=args.model_size,
        epochs=args.epochs,
        batch=args.batch,
        resume_path=args.resume,
    )
    print(f"\n训练完成，模型: {out}")


if __name__ == "__main__":
    main()
