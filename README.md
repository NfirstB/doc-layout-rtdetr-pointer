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
pip install ultralytics torch pdfplumber pyyaml pillow
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

## License

MIT
