# doc-layout-analyzer

> 文档布局分析 + 阅读顺序推理 Pipeline

功能：输入 PDF → 检测标题/正文/图片/表格/公式 → 输出阅读顺序 JSON

---

## 项目结构

```
doc-layout-analyzer/
├── src/
│   ├── pdf_parser.py       # PDF 解析（文本坐标/图片提取）
│   ├── reading_order.py    # 阅读顺序推理（2D坐标+栏位分析）
│   └── pipeline.py         # 完整 Pipeline（解析→检测→排序）
├── scripts/
│   ├── train_yolo.py       # YOLO 模型训练
│   ├── label_converter.py  # Label-Studio ↔ YOLO 格式互转
│   └── run_tmux.py         # tmux 任务启动器（所有>5min任务）
├── data/                    # 标注数据
├── models/                  # 训练好的模型
├── configs/                 # 配置文件
└── README.md
```

---

## 快速开始

### 1. 安装依赖

```bash
pip install ultralytics torch pymupdf pillow numpy
```

### 2. 准备标注数据

使用 Label-Studio 标注，导出 JSON 后转换为 YOLO 格式：

```bash
python scripts/label_converter.py \
    --input ./ls_export.json \
    --output ./data/yolo_dataset/ \
    --to-yolo
```

### 3. 训练模型（>5分钟必须用 tmux）

```bash
# 在 tmux 里启动训练（立即返回，不阻塞）
python scripts/run_tmux.py train \
    --data ./data/yolo_dataset/ \
    --epochs 100 \
    --batch 16 \
    --size m \
    --name layout_train

# 查看训练进度
tmux attach -t layout_train
# 或看日志
tail -f /tmp/tmux_layout_train.log

# 查看所有 tmux session
python scripts/run_tmux.py status

# 关闭 session
tmux kill-session -t layout_train
```

### 4. 批量推理

```bash
python scripts/run_tmux.py infer \
    --pdf-dir ./pdfs/ \
    --output ./output/ \
    --model ./models/layout_yolov8m.pt
```

---

## 标签类别（8类）

| ID | 标签 | 说明 |
|----|------|------|
| 0 | title | 文档/页面标题 |
| 1 | header | 章节标题 |
| 2 | body | 正文段落 |
| 3 | figure | 图片 |
| 4 | caption | 图片/表格标题 |
| 5 | table | 表格 |
| 6 | footnote | 脚注 |
| 7 | equation | 公式 |

---

## 阅读顺序推理算法

```
1. 检测栏数（1栏/2栏/3栏）
   → 分析元素左边界分布，找双峰分隔点

2. 分离跨栏元素（标题/图片/表格）
   → 宽度 > 单栏 80% 的视为跨栏

3. 普通元素按行分组
   → 用 y 坐标聚类估算行高
   → 每个元素分配 (row, col)

4. 排序：先 row（从上到下），再 col（从左到右）
   → 跨栏元素按最近上方普通元素位置插入

5. 输出：按阅读顺序排列的 JSON
```

---

## Label-Studio 部署

```bash
# 本地启动 Label-Studio
pip install label-studio
label-studio start

# 生成标注配置
python scripts/label_converter.py --gen-config label_config.xml
```

在 Label-Studio Web UI 中导入 `label_config.xml` 即可开始标注。

---

## 配套项目

| 项目 | 说明 |
|------|------|
| [mmkg-agent](https://github.com/NfirstB/mmkg-agent) | 论文 → 知识图谱（用这个模块做布局分析输入）|
| [mmkg-papers](https://github.com/NfirstB/mmkg-papers) | Transformer 论文调研 |

---

*最后更新：2026-03-27*
