#!/usr/bin/env python3
"""
6层 Transformer 编码器 + 几何偏置 (Geometric Bias)

论文思路：
- Relation-DETR 的几何偏置机制：显式建模元素之间的空间几何关系
- 比如"A在B的上方"、"C在D的右侧"这种空间关系

实现方式：
- 在标准自注意力中加入几何偏置项
- 几何偏置基于两个元素边界框的相对位置关系
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Linear, LayerNorm, Dropout

# 从 position_encoding 导入
from .position_encoding import Sinusoidal2DPositionalEncoding, CategoryEmbedding


class GeometricBias(nn.Module):
    """
    几何偏置计算模块

    根据两个元素的边界框，计算它们之间的空间关系作为偏置。

    关系包括：
    - 垂直关系：element i 是否在 element j 的上方/下方
    - 水平关系：element i 是否在 element j 的左侧/右侧
    - 包含关系：element i 是否被 element j 包含

    输出：N×N 的几何偏置矩阵（加到注意力分数上）
    """
    def __init__(self, temperature: float = 10.0):
        super().__init__()
        self.temperature = temperature

    def forward(self, bbox: torch.Tensor) -> torch.Tensor:
        """
        Args:
            bbox: (N, 4)，归一化边界框 [x0, y0, x1, y1]
        Returns:
            bias: (N, N)，几何偏置矩阵
        """
        N = bbox.shape[0]

        x0, y0, x1, y1 = bbox[:, 0], bbox[:, 1], bbox[:, 2], bbox[:, 3]
        cx = (x0 + x1) / 2
        cy = (y0 + y1) / 2

        dx = cx.unsqueeze(1) - cx.unsqueeze(0)
        dy = cy.unsqueeze(1) - cy.unsqueeze(0)

        horiz = torch.sign(dx)
        vert = torch.sign(dy)

        dist = torch.sqrt(dx ** 2 + dy ** 2 + 1e-6)
        dist = dist / (dist.max() + 1e-6)

        bias = (torch.abs(horiz) + torch.abs(vert)) * torch.exp(-dist * 3)
        bias = bias * (1 - torch.eye(N, device=bias.device))
        bias = bias / (bias.max() + 1e-6) * self.temperature

        return bias


class TransformerEncoderLayer(nn.Module):
    """
    单层 Transformer 编码器（带几何偏置）

    包含：
    - 多头自注意力（带几何偏置）
    - 前馈网络 (FFN)
    - 残差连接 + LayerNorm
    """
    def __init__(self, d_model: int, nhead: int = 8, dim_feedforward: int = 2048,
                 dropout: float = 0.1):
        super().__init__()

        self.self_attn = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=True
        )

        self.linear1 = Linear(d_model, dim_feedforward)
        self.dropout = Dropout(dropout)
        self.linear2 = Linear(dim_feedforward, d_model)

        self.norm1 = LayerNorm(d_model)
        self.norm2 = LayerNorm(d_model)
        self.dropout1 = Dropout(dropout)
        self.dropout2 = Dropout(dropout)

        self.activation = nn.ReLU()
        self.geom_bias = GeometricBias(temperature=5.0)

    def forward(self, src: torch.Tensor, bbox: torch.Tensor) -> torch.Tensor:
        geom_bias = self.geom_bias(bbox)
        d_k = src.shape[-1] ** 0.5

        q = src
        k = src
        v = src

        attn_weights = torch.matmul(q, k.transpose(-2, -1)) / d_k
        attn_weights = attn_weights + geom_bias
        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_weights = self.dropout1(attn_weights)

        output = torch.matmul(attn_weights, v)
        output = self.norm1(output + src)
        output2 = self.linear2(self.dropout(self.activation(self.linear1(output))))
        output = self.norm2(output + self.dropout2(output2))

        return output


class TransformerEncoder(nn.Module):
    """
    6层 Transformer 编码器
    """
    def __init__(self, d_model: int = 256, nhead: int = 8,
                 num_layers: int = 6, dim_feedforward: int = 2048,
                 dropout: float = 0.1):
        super().__init__()

        self.layers = nn.ModuleList([
            TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout)
            for _ in range(num_layers)
        ])
        self.num_layers = num_layers

    def forward(self, src: torch.Tensor, bbox: torch.Tensor) -> torch.Tensor:
        output = src
        for layer in self.layers:
            output = layer(output, bbox)
        return output


class PairwiseRelationHead(nn.Module):
    """
    成对关系头 (Pairwise Relation Head)

    论文思路：
    - 把元素特征投影为 query 向量和 key 向量
    - 计算双线性相似度得到 N×N 的相似度矩阵
    - 矩阵中第 i 行第 j 列的值 = "元素 i 在元素 j 前面读" 的概率

    公式：
        S_ij = query(element_i)^T * key(element_j) / sqrt(d_k)
    """
    def __init__(self, d_model: int, d_k: int = 128):
        super().__init__()

        self.query_proj = nn.Linear(d_model, d_k)
        self.key_proj = nn.Linear(d_model, d_k)
        self.scale = math.sqrt(d_k)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        query = self.query_proj(features)
        key = self.key_proj(features)
        relation = torch.matmul(query, key.transpose(-2, -1)) / self.scale
        return relation


class ElementFeatureEncoder(nn.Module):
    """
    元素特征编码器
    """
    def __init__(self, num_classes: int, d_model: int = 256, bbox_embed_dim: int = 128):
        super().__init__()
        self.num_classes = num_classes
        self.d_model = d_model

        self.bbox_embed = nn.Sequential(
            nn.Linear(4, bbox_embed_dim),
            nn.ReLU(),
            nn.Linear(bbox_embed_dim, bbox_embed_dim),
            nn.ReLU(),
        )

        self.pos_encoding = Sinusoidal2DPositionalEncoding(d_model)
        self.center_embed = nn.Sequential(
            nn.Linear(2, d_model // 4),
            nn.ReLU(),
        )
        self.size_embed = nn.Sequential(
            nn.Linear(2, d_model // 4),
            nn.ReLU(),
        )
        self.cat_embedding = CategoryEmbedding(num_classes, d_model // 2)

        self.fusion = nn.Sequential(
            nn.Linear(bbox_embed_dim + d_model // 2 + d_model // 2, d_model),
            nn.ReLU(),
            nn.Dropout(0.1),
        )

    def forward(self, bbox: torch.Tensor, category: torch.Tensor) -> torch.Tensor:
        bbox_feat = self.bbox_embed(bbox)
        cx = (bbox[:, 0] + bbox[:, 2]) / 2
        cy = (bbox[:, 1] + bbox[:, 3]) / 2
        center_feat = self.center_embed(torch.stack([cx, cy], dim=-1))
        w = bbox[:, 2] - bbox[:, 0]
        h = bbox[:, 3] - bbox[:, 1]
        size_feat = self.size_embed(torch.stack([w, h], dim=-1))
        x0_norm = bbox[:, 0]
        y0_norm = bbox[:, 1]
        pos_feat = self.pos_encoding(x0_norm, y0_norm)
        cat_feat = self.cat_embedding(category)
        combined = torch.cat([bbox_feat, cat_feat, pos_feat[:, :self.d_model // 2]], dim=-1)
        features = self.fusion(combined)
        return features


class PointerNetworkReadingOrder(nn.Module):
    """
    指针网络阅读顺序模型

    完整流程：
    1. ElementFeatureEncoder：编码检测框 + 类别 → d_model 维向量
    2. TransformerEncoder（6层）：加入几何偏置，编码元素间关系
    3. PairwiseRelationHead：计算 N×N 相似度矩阵

    输出：
    - relation_matrix: (N, N)，第 (i,j) 元素 = 元素 i 在元素 j 之前的概率
    """
    def __init__(self, num_classes: int = 8, d_model: int = 256,
                 nhead: int = 8, num_layers: int = 6,
                 d_k: int = 128):
        super().__init__()

        self.num_classes = num_classes
        self.d_model = d_model

        self.encoder = ElementFeatureEncoder(num_classes, d_model)
        self.transformer = TransformerEncoder(
            d_model=d_model,
            nhead=nhead,
            num_layers=num_layers,
        )
        self.relation_head = PairwiseRelationHead(d_model, d_k)

    def forward(self, bbox: torch.Tensor, category: torch.Tensor):
        features = self.encoder(bbox, category)
        encoded = self.transformer(features, bbox)
        relation_matrix = self.relation_head(encoded)
        return relation_matrix


if __name__ == "__main__":
    # 测试
    import numpy as np

    N = 5
    num_classes = 8
    d_model = 128

    model = PointerNetworkReadingOrder(
        num_classes=num_classes,
        d_model=d_model,
        nhead=4,
        num_layers=6,
    )
    model.eval()

    bbox = torch.rand(N, 4)
    category = torch.randint(0, num_classes, (N,))

    with torch.no_grad():
        relation = model(bbox, category)

    print(f"Input: bbox {bbox.shape}, category {category.shape}")
    print(f"Relation matrix: {relation.shape}")
    print(f"Test passed!")
