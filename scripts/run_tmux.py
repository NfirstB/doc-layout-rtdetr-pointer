#!/usr/bin/env python3
"""
tmux 训练启动器 —— 所有 >5 分钟的任务都通过这个脚本在 tmux 里跑
"""
import argparse
import subprocess
import sys
import os
from pathlib import Path

ROOT = Path(__file__).parent.parent


def run_in_tmux(session_name: str, command: str, wait: bool = True):
    """在 tmux session 中运行命令"""
    # 检查 session 是否已存在
    check = subprocess.run(
        ["tmux", "has-session", "-t", session_name],
        capture_output=True
    )
    if check.returncode == 0:
        print(f"Session '{session_name}' 已存在，附加...")
        subprocess.run(["tmux", "attach", "-t", session_name])
        return

    # 创建新 session，后台运行
    if wait:
        # 前台模式（等待完成）
        proc = subprocess.run(
            ["tmux", "new-session", "-d", "-s", session_name, command],
            cwd=str(ROOT)
        )
    else:
        # 后台模式（立即返回）
        proc = subprocess.run(
            ["tmux", "new-session", "-d", "-s", session_name, f"cd {ROOT} && {command}"],
            cwd=str(ROOT)
        )

    if proc.returncode == 0:
        print(f"✓ 已启动 tmux session: {session_name}")
        print(f"  查看日志: tmux capture-pane -t {session_name} -p")
        print(f"  附加会话: tmux attach -t {session_name}")
        print(f"  等待完成: tail -f /tmp/tmux_{session_name}.log")
    else:
        print(f"✗ 启动失败: {proc.stderr.decode()}")


def main():
    parser = argparse.ArgumentParser(description="tmux 任务启动器")
    sub = parser.add_subparsers(dest="cmd")

    # 训练命令
    p_train = sub.add_parser("train", help="启动 YOLO 训练（tmux）")
    p_train.add_argument("--data", default="./data/annotated", help="标注数据目录")
    p_train.add_argument("--epochs", type=int, default=100, help="训练轮数")
    p_train.add_argument("--batch", type=int, default=16, help="批次大小")
    p_train.add_argument("--size", default="m", choices=["n","s","m","l","x"], help="模型大小")
    p_train.add_argument("--detach", action="store_true", help="后台运行")

    # 推理命令
    p_infer = sub.add_parser("infer", help="批量推理（tmux）")
    p_infer.add_argument("--pdf-dir", required=True, help="PDF 目录")
    p_infer.add_argument("--output", default="./output", help="输出目录")
    p_infer.add_argument("--model", default=None, help="模型路径")

    # 格式转换
    p_convert = sub.add_parser("convert", help="标注格式转换")
    p_convert.add_argument("--input", required=True, help="输入")
    p_convert.add_argument("--output", required=True, help="输出")
    p_convert.add_argument("--to-yolo", action="store_true")
    p_convert.add_argument("--to-ls", action="store_true")

    # 安装依赖
    p_install = sub.add_parser("install", help="安装依赖")

    args = parser.parse_args()

    if args.cmd == "train":
        cmd = (
            f"python3 scripts/train_yolo.py "
            f"--data-dir {args.data} "
            f"--epochs {args.epochs} "
            f"--batch {args.batch} "
            f"--model-size {args.size}"
        )
        run_in_tmux("layout_train", cmd, wait=not args.detach)

    elif args.cmd == "infer":
        cmd = (
            f"python3 -m src.pipeline "
            f"--pdf-dir {args.pdf_dir} "
            f"--output {args.output}"
        )
        if args.model:
            cmd += f" --model {args.model}"
        run_in_tmux("layout_infer", cmd, wait=True)

    elif args.cmd == "convert":
        cmd = f"python3 scripts/label_converter.py --input {args.input} --output {args.output}"
        if args.to_yolo: cmd += " --to-yolo"
        if args.to_ls: cmd += " --to-ls"
        subprocess.run(cmd, shell=True, cwd=str(ROOT))

    elif args.cmd == "install":
        pkgs = ["ultralytics", "torch", "pymupdf", "label-studio"]
        print(f"安装: {pkgs}")
        subprocess.run([sys.executable, "-m", "pip", "install"] + pkgs, check=True)
        print("✓ 安装完成")

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
