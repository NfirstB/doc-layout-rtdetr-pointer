# DocLayout-RTDETR-Pointer

**RT-DETR / YOLOv8 检测 + Pointer Network 阅读顺序** 文档布局分析 pipeline。

## 架构

```
PDF → 检测模型(RT-DETR/YOLO) → 边界框+类别
                                      ↓
                          Pointer Network → N×N相似度矩阵 → 阅读顺序
```

### 子模块

| 模块 | 文件 | 说明 |
|------|------|------|
| 2D位置编码 | `src/pointer_network/position_encoding.py` | 正弦编码 x,y 坐标 + 类别 embedding |
| Transformer编码器 | `src/pointer_network/transformer_encoder.py` | 6层 Transformer + 几何偏置 |
| 解码算法 | `src/pointer_network/decoding.py` | 赢累积解码，恢复拓扑一致的阅读顺序 |
| 训练接口 | `src/pointer_network/train_pointer.py` | Pointer Network 训练 + 伪标签生成 |
| 检测脚本 | `scripts/infer_rtdetr_pointer.py` | 完整 pipeline 推理 |

## 模型下载

### 检测模型（YOLO）
```bash
# YOLOv8n (轻量, mAP@0.5≈0.88)
wget https://github.com/NfirstB/doc-layout-rtdetr-pointer/releases/download/v1.0/layout_yolov8n.pt

# YOLOv8m (中等)
wget https://github.com/NfirstB/doc-layout-rtdetr-pointer/releases/download/v1.0/layout_yolov8m.pt
```

### Pointer Network（阅读顺序）
```bash
# 伪标签训练的初始版本（待标注数据后重新训练）
wget https://github.com/NfirstB/doc-layout-rtdetr-pointer/releases/download/v1.0/pointer_network_pseudo.pt
```

## 使用方法

### 完整推理（检测 + 阅读顺序）
```bash
python scripts/infer_rtdetr_pointer.py \
    --detector models/layout_yolov8n.pt \
    --pointer models/pointer_network_pseudo.pt \
    --pdf paper.pdf \
    --page 0 \
    --output result.jpg
```

### RT-DETR 训练
```bash
python scripts/train_rtdetr.py \
    --data-dir /path/to/yolo_dataset \
    --epochs 50 \
    --batch 8 \
    --model-size l
```

### YOLOv8 训练
```bash
python scripts/train_yolo.py \
    --data-dir /path/to/yolo_dataset \
    --epochs 50 \
    --batch 16 \
    --model-size n
```

### PubLayNet 数据集下载
```bash
python scripts/download_publaynet.py \
    --output ./publaynet_yolo \
    --num-train 500 \
    --num-val 50
```

## 依赖

```bash
pip install ultralytics torch pdfplumber pyyaml pillow pypdfium2 httpx
```

## 自动标注流水线（Auto Label Pipeline）

用 **专家模型 + 大模型** 实现低成本高精度自动标注，替代纯人工标注。

### 架构

```
PDF 页面
    ↓
Step 1: 专家模型（YOLO 检测）
    生成伪标签：bbox + class + conf
    ↓
Step 2: VLM 精修（MiniMax VL）
    图像 + 伪标签 → Prompt → 大模型修正错误
    ↓
Step 3: 幻觉过滤
    规则过滤：几何约束 + 类别规则 + 重叠检测
    ↓
高质量标注结果
```

### 使用方法

```python
from auto_label import AutoLabelPipeline

pipeline = AutoLabelPipeline()

# 单页
result = pipeline.run("/path/to/paper.pdf", page_num=0, dpi=150)

print(result["kept"])        # 最终标注元素
print(result["stats"])        # 统计信息
result["page_image"].save("annotated.jpg")  # 可视化

# 批量
results = pipeline.run_batch([
    "/path/to/paper1.pdf",
    "/path/to/paper2.pdf",
], pages_per_pdf=[0, 1], output_dir="./outputs")
```

### CLI 用法

```bash
# 完整流水线
python -m auto_label.pipeline /path/to/paper.pdf --page 0 --dpi 150 -o ./outputs

# 跳过 VLM（仅专家模型 + 过滤）
python -m auto_label.pipeline /path/to/paper.pdf --skip-vlm

# 跳过过滤（仅专家模型 + VLM）
python -m auto_label.pipeline /path/to/paper.pdf --skip-filter
```

### 子模块说明

| 模块 | 文件 | 说明 |
|------|------|------|
| 主管道 | `auto_label/pipeline.py` | 调度 Step 1→2→3 |
| 专家检测 | `auto_label/expert_layout.py` | YOLO 检测生成伪标签 |
| VLM 精修 | `auto_label/vlm_refine.py` | MiniMax VL API 修正伪标签 |
| 幻觉过滤 | `auto_label/filter_hallucination.py` | 规则过滤假阳性 |
| Prompt 模板 | `auto_label/prompt_templates.py` | VLM Prompt 工程 |

### API Key 配置

VLM 精修步骤需要 MiniMax API Key（在 `vlm_refine.py` 中配置）:

```python
MINIMAX_API_KEY = "your_key_here"
```

### 依赖

```bash
pip install ultralytics pypdfium2 httpx pillow
```

## 数据集格式

YOLO 格式：
```
dataset/
├── images/
│   ├── train/
│   └── val/
├── labels/
│   ├── train/
│   └── val/
└── data.yaml
```

`data.yaml` 示例：
```yaml
path: ./dataset
train: images/train
val: images/val
nc: 8
names: ['title', 'header', 'body', 'figure', 'caption', 'table', 'footnote', 'equation']
```

## Pointer Network 架构

- **输入**: 边界框 (N, 4) + 类别 (N,) → d_model=256 维向量
- **编码器**: 6层 Transformer + Relation-DETR 几何偏置
- **关系头**: 成对双线性相似度 → N×N 矩阵（S_ij = "i在j前读"的概率）
- **解码**: 赢累积算法，保证拓扑一致性

## 论文

如果这个项目对你有帮助，请引用：
- Buffer of Thoughts: arXiv:2406.04271
- ReasonFlux: arXiv:2502.06772

## 更新日志

### v1.1 (2026-03-30)
- **修复可视化黑图 bug**: `visualize()` 函数使用 `alpha_composite` 合并 RGBA 层后再转 RGB，解决了透明区域像素变黑的问题
- **框外标签**: 类别标签从框内移至框外下方，避免遮挡 PDF 文字
- **透明边框**: 检测框改为全透明填充 + 细边框，完整保留 PDF 可读性

### v1.0 (2026-03-25)
- 初始版本：RT-DETR/YOLO 检测 + Pointer Network 阅读顺序

## License

MIT
