#!/usr/bin/env python3
"""
tmux 任务启动器 —— 所有 >5 分钟的任务必须用这个在 tmux 里跑

用法：
  python scripts/run_tmux.py train --epochs 100 --batch 16
  python scripts/run_tmux.py infer --pdf-dir ./pdfs/
  python scripts/run_tmux.py status
  tmux attach -t layout_train      # 查看训练进度
  tail -f /tmp/tmux_layout_train.log  # 看日志
"""
import argparse
import subprocess
import sys
import os
import atexit
from pathlib import Path

ROOT = Path(__file__).parent.parent


def get_free_session_name(base: str) -> str:
    """找一个空闲的 session 名"""
    for i in range(20):
        name = f"{base}_{i}" if i > 0 else base
        r = subprocess.run(
            ["tmux", "has-session", "-t", name],
            capture_output=True)
        if r.returncode != 0:
            return name
    return f"{base}_{os.getpid()}"


def run_in_tmux(session: str, work_dir: Path,
                script: str, python_args: list):
    """
    在 tmux session 里运行 Python 脚本，日志写到 /tmp/tmux_<session>.log

    session    : tmux session 名
    work_dir   : 工作目录
    script     : 脚本路径（相对于 work_dir）
    python_args: 传给脚本的参数列表
    """
    log_file = f"/tmp/tmux_{session}.log"

    # 检查是否已存在
    r = subprocess.run(
        ["tmux", "has-session", "-t", session],
        capture_output=True)
    if r.returncode == 0:
        print(f"Session '{session}' 已存在，直接附加...")
        subprocess.run(["tmux", "attach", "-t", session])
        return

    # 构造命令：用 exec bash -l 确保环境干净
    py_cmd = " ".join(python_args)
    tmux_cmd = (
        f"exec bash -l -c '"
        f"cd {work_dir} && "
        f"python3 -u {script} {py_cmd} "
        f"> {log_file} 2>&1'"   # -u 无缓冲，日志实时
    )

    r = subprocess.run(
        ["tmux", "new-session", "-d", "-s", session, tmux_cmd],
        capture_output=True)

    if r.returncode != 0:
        print(f"✗ 启动 tmux 失败: {r.stderr.decode()}")
        return

    print(f"✓ tmux session 已启动: {session}")
    print(f"  日志: tail -f {log_file}")
    print(f"  附加: tmux attach -t {session}")
    print(f"  状态: python scripts/run_tmux.py status")

    def cleanup():
        print(f"\nSession '{session}' 仍在运行:")
        print(f"  日志: tail -f {log_file}")

    atexit.register(cleanup)


def main():
    parser = argparse.ArgumentParser(description="tmux 任务启动器")
    sub = parser.add_subparsers(dest="cmd", required=True)

    # ── train ────────────────────────────────────────────────────────────
    p = sub.add_parser("train", help="启动 YOLO 训练（tmux）")
    p.add_argument("--data-dir", default=str(ROOT / "data" / "yolo_dataset"))
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--model-size", default="m",
                   choices=["n", "s", "m", "l", "x"])
    p.add_argument("--name", default="layout_train",
                   help="tmux session 名")
    p.add_argument("--check", action="store_true",
                   help="仅检查依赖")

    # ── infer ────────────────────────────────────────────────────────────
    p = sub.add_parser("infer", help="批量 PDF 推理（tmux）")
    p.add_argument("--pdf-dir", required=True)
    p.add_argument("--output", default=str(ROOT / "output"))
    p.add_argument("--model", default=None)
    p.add_argument("--name", default="layout_infer")

    # ── status ────────────────────────────────────────────────────────────
    sub.add_parser("status", help="查看所有 tmux session")

    # ── install ───────────────────────────────────────────────────────────
    sub.add_parser("install", help="安装 Python 依赖")

    args = parser.parse_args()

    # ── status ────────────────────────────────────────────────────────────
    if args.cmd == "status":
        r = subprocess.run(
            ["tmux", "list-sessions", "-F", "#{session_name}"],
            capture_output=True, text=True)
        if r.returncode == 0 and r.stdout.strip():
            print("活跃 sessions:")
            for line in r.stdout.strip().split("\n"):
                print(f"  • {line}")
        else:
            print("没有活跃的 tmux sessions")
        return

    # ── install ───────────────────────────────────────────────────────────
    if args.cmd == "install":
        pkgs = [
            "ultralytics>=8.0",
            "torch>=2.0",
            "pymupdf",
            "pillow",
            "numpy",
        ]
        cmd = [sys.executable, "-m", "pip", "install"] + pkgs
        print(f"安装: {' '.join(pkgs)}")
        subprocess.run(cmd, check=True)
        print("✓ 安装完成")
        return

    # ── train ────────────────────────────────────────────────────────────
    if args.cmd == "train":
        session = args.name
        log_file = f"/tmp/tmux_{session}.log"

        # 先检查依赖
        if args.check:
            sys.path.insert(0, str(ROOT))
            from train_yolo import check_dependencies
            check_dependencies()
            return

        script = "scripts/train_yolo.py"
        python_args = [
            "--data-dir", str(ROOT / args.data_dir) if args.data_dir.startswith("./") else args.data_dir,
            "--epochs", str(args.epochs),
            "--batch", str(args.batch),
            "--model-size", args.model_size,
        ]

        run_in_tmux(session, ROOT, script, python_args)
        print(f"\n训练已提交到 tmux（{args.epochs} epochs）")
        print("等几秒后查看进度：")
        print(f"  tail -f {log_file}")
        return

    # ── infer ────────────────────────────────────────────────────────────
    if args.cmd == "infer":
        session = args.name
        script = "src/pipeline.py"
        python_args = [
            "--pdf-dir", args.pdf_dir,
            "--output", args.output,
        ]
        if args.model:
            python_args += ["--model", args.model]

        run_in_tmux(session, ROOT, script, python_args)
        return


if __name__ == "__main__":
    main()
