#!/usr/bin/env python3
"""
tmux 任务启动器 —— 所有 >5 分钟的任务通过这个在 tmux 里跑
用法:
  python scripts/run_tmux.py train --epochs 100 --batch 16
  python scripts/run_tmux.py infer --pdf-dir ./pdfs/
  tmux attach -t layout_train       # 查看训练进度
  tmux capture-pane -t layout_train -p  # 查看最后输出
"""
import argparse
import subprocess
import sys
import os
import json
import atexit
from pathlib import Path

ROOT = Path(__file__).parent.parent


def get_free_session_name(prefix: str) -> str:
    """找一个空闲的 session 名"""
    for i in range(10):
        name = f"{prefix}_{i}" if i > 0 else prefix
        r = subprocess.run(["tmux", "has-session", "-t", name],
                          capture_output=True)
        if r.returncode != 0:
            return name
    return f"{prefix}_{os.getpid()}"


def get_tmux_pid(session_name: str) -> str:
    """获取 tmux session 的 PID"""
    r = subprocess.run(
        ["tmux", "list-panes", "-t", session_name, "-F", "#{pane_pid}"],
        capture_output=True, text=True
    )
    if r.returncode == 0:
        return r.stdout.strip()
    return ""


def run_in_tmux(session_name: str, script_path: str,
                python_args: list, log_file: str):
    """
    在 tmux session 里运行 Python 脚本
    session_name : tmux session 名
    script_path  : 要运行的 .py 文件（相对于 ROOT）
    python_args  : 传给脚本的参数
    log_file     : 输出日志文件
    """
    # 检查 session 是否已存在
    r = subprocess.run(["tmux", "has-session", "-t", session_name],
                      capture_output=True)
    if r.returncode == 0:
        print(f"Session '{session_name}' 已存在，附加...")
        subprocess.run(["tmux", "attach", "-t", session_name])
        return

    # 构造 tmux 命令：用 bash 登录式 shell 确保环境完整
    abs_script = str((ROOT / script_path).resolve())
    py_args_str = " ".join(python_args)
    cmd = (
        f"exec bash -l -c '"
        f"cd {ROOT} && "
        f"python3 {abs_script} {py_args_str} "
        f"> {log_file} 2>&1'"
    )

    r = subprocess.run(["tmux", "new-session", "-d", "-s", session_name, cmd],
                      capture_output=True)
    if r.returncode != 0:
        err = r.stderr.decode()
        print(f"✗ 启动 tmux 失败: {err}")
        return

    # 注册退出提示
    def cleanup():
        print(f"\nSession '{session_name}' 还在运行:")
        print(f"  查看日志: tail -f {log_file}")
        print(f"  附加会话: tmux attach -t {session_name}")
        print(f"  关闭会话: tmux kill-session -t {session_name}")

    atexit.register(cleanup)
    print(f"✓ 已启动 tmux session: {session_name}")
    print(f"  日志: tail -f {log_file}")
    print(f"  附加: tmux attach -t {session_name}")


def main():
    parser = argparse.ArgumentParser(description="tmux 任务启动器")
    sub = parser.add_subparsers(dest="cmd", required=True)

    # ── train ────────────────────────────────────────────────
    p = sub.add_parser("train", help="启动 YOLO 训练")
    p.add_argument("--data", default="./data/yolo_dataset",
                   help="YOLO 数据集目录")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--size", default="m",
                   choices=["n", "s", "m", "l", "x"],
                   help="YOLO 模型大小")
    p.add_argument("--name", default="layout_train",
                   help="tmux session 名")

    # ── infer ────────────────────────────────────────────────
    p = sub.add_parser("infer", help="批量 PDF 推理")
    p.add_argument("--pdf-dir", required=True)
    p.add_argument("--output", default="./output")
    p.add_argument("--model", default=None)
    p.add_argument("--name", default="layout_infer",
                   help="tmux session 名")

    # ── install ───────────────────────────────────────────────
    p = sub.add_parser("install", help="安装依赖")

    # ── status ────────────────────────────────────────────────
    p = sub.add_parser("status", help="查看所有 session 状态")

    args = parser.parse_args()

    # ── install ──────────────────────────────────────────────
    if args.cmd == "install":
        pkgs = ["ultralytics>=8.0", "torch", "pymupdf", "pillow", "numpy"]
        print(f"安装: {' '.join(pkgs)}")
        subprocess.run([sys.executable, "-m", "pip", "install"] + pkgs, check=True)
        print("✓ 安装完成")
        return

    # ── status ────────────────────────────────────────────────
    if args.cmd == "status":
        r = subprocess.run(["tmux", "list-sessions", "-F", "#{session_name}"],
                          capture_output=True, text=True)
        if r.returncode == 0 and r.stdout.strip():
            print("活跃 tmux sessions:")
            for line in r.stdout.strip().split("\n"):
                print(f"  • {line}")
        else:
            print("没有活跃的 tmux sessions")
        return

    # ── train ────────────────────────────────────────────────
    if args.cmd == "train":
        session = args.name
        log_file = f"/tmp/tmux_{session}.log"
        script = "scripts/train_yolo.py"

        # 将参数写成 flagfile，脚本读取（避免 shell 转义问题）
        flagfile = f"/tmp/tmux_{session}_flags.json"
        flags = {
            "data_dir": args.data,
            "epochs": args.epochs,
            "batch": args.batch,
            "model_size": args.size,
        }
        with open(flagfile, "w") as f:
            json.dump(flags, f)

        python_args = [f"--flagfile={flagfile}"]
        run_in_tmux(session, script, python_args, log_file)
        print(f"\n训练已后台启动，{args.epochs} epochs 预计 30分钟-数小时")
        return

    # ── infer ────────────────────────────────────────────────
    if args.cmd == "infer":
        session = args.name
        log_file = f"/tmp/tmux_{session}.log"
        script = "src/pipeline.py"

        python_args = [
            "--pdf-dir", args.pdf_dir,
            "--output", args.output,
        ]
        if args.model:
            python_args += ["--model", args.model]

        run_in_tmux(session, script, python_args, log_file)
        return


if __name__ == "__main__":
    main()
