#!/usr/bin/env python3
"""
2D Absolute Position Encoding + Category Embedding for Pointer Network

论文思路：
- 用 (x, y) 坐标表示元素在文档里的位置，转化为向量
- 让模型知道"这个元素在页面的左上角"
- 结合类别标签 embedding，形成统一的特征向量
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class Sinusoidal2DPositionalEncoding(nn.Module):
    """
    2D 绝对位置编码 (Sinusoidal 2D Positional Encoding)

    对每个元素的 (x, y) 坐标分别做正弦/余弦编码，
    然后 concat 得到 d_model 维的位置向量。

    公式：
        PE_x(p, 2i)   = sin(p_x / 10000^(2i/d_model))
        PE_x(p, 2i+1) = cos(p_x / 10000^(2i/d_model))
        (y 坐标同理)
    """
    def __init__(self, d_model: int):
        super().__init__()
        assert d_model % 2 == 0, "d_model must be even"
        self.d_model = d_model

        # 频率倒数（用于生成不同波长的正弦曲线）
        inv_freq = 1.0 / (10000 ** (torch.arange(0, d_model // 2, dtype=torch.float) / (d_model // 2)))
        self.register_buffer("inv_freq", inv_freq)

    def forward(self, x_norm: torch.Tensor, y_norm: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x_norm: Tensor of shape (N,)，x坐标归一化到 [0, 1]
            y_norm: Tensor of shape (N,)，y坐标归一化到 [0, 1]
        Returns:
            pos_encoding: Tensor of shape (N, d_model)
        """
        # x encoding: (N, d_model//2)
        x_enc_x = torch.sin(x_norm.unsqueeze(-1) * self.inv_freq)   # sin part
        x_enc_y = torch.cos(x_norm.unsqueeze(-1) * self.inv_freq)  # cos part

        # y encoding: (N, d_model//2)
        y_enc_x = torch.sin(y_norm.unsqueeze(-1) * self.inv_freq)
        y_enc_y = torch.cos(y_norm.unsqueeze(-1) * self.inv_freq)

        # concat: (N, d_model)
        pos_encoding = torch.cat([x_enc_x, x_enc_y, y_enc_x, y_enc_y], dim=-1)
        return pos_encoding


class CategoryEmbedding(nn.Module):
    """
    类别标签 embedding

    每个类别（title/header/body/figure/caption/table/footnote/equation）
    映射到一个 d_model 维向量。
    """
    def __init__(self, num_classes: int, d_model: int):
        super().__init__()
        self.embedding = nn.Embedding(num_classes, d_model)

    def forward(self, category_ids: torch.Tensor) -> torch.Tensor:
        """
        Args:
            category_ids: Tensor of shape (N,)，类别 ID（int）
        Returns:
            cat_embedding: Tensor of shape (N, d_model)
        """
        return self.embedding(category_ids)


class ElementFeatureEncoder(nn.Module):
    """
    元素特征编码器

    将检测到的元素（边界框 + 类别）编码为统一向量表示。

    输入：
        - bbox: (N, 4)，归一化的 [x0, y0, x1, y1]
        - category: (N,)，类别 ID

    输出：
        - element_features: (N, d_model)
    """
    def __init__(self, num_classes: int, d_model: int = 256, bbox_embed_dim: int = 128):
        super().__init__()
        self.num_classes = num_classes
        self.d_model = d_model

        # 边界框特征提取：4个坐标值 → bbox_embed_dim
        self.bbox_embed = nn.Sequential(
            nn.Linear(4, bbox_embed_dim),
            nn.ReLU(),
            nn.Linear(bbox_embed_dim, bbox_embed_dim),
            nn.ReLU(),
        )

        # 2D 位置编码
        self.pos_encoding = Sinusoidal2DPositionalEncoding(d_model)

        # 中心坐标（用于位置编码）
        self.center_embed = nn.Sequential(
            nn.Linear(2, d_model // 4),
            nn.ReLU(),
        )

        # 宽高信息
        self.size_embed = nn.Sequential(
            nn.Linear(2, d_model // 4),
            nn.ReLU(),
        )

        # 类别 embedding
        self.cat_embedding = CategoryEmbedding(num_classes, d_model // 2)

        # 融合层
        self.fusion = nn.Sequential(
            nn.Linear(bbox_embed_dim + d_model // 2 + d_model // 2, d_model),
            nn.ReLU(),
            nn.Dropout(0.1),
        )

    def forward(self, bbox: torch.Tensor, category: torch.Tensor) -> torch.Tensor:
        """
        Args:
            bbox: (N, 4)，归一化边界框 [x0, y0, x1, y1] ∈ [0, 1]
            category: (N,)，类别 ID
        Returns:
            features: (N, d_model)
        """
        # 边界框特征
        bbox_feat = self.bbox_embed(bbox)  # (N, bbox_embed_dim)

        # 中心坐标
        cx = (bbox[:, 0] + bbox[:, 2]) / 2  # (N,)
        cy = (bbox[:, 1] + bbox[:, 3]) / 2  # (N,)
        center_feat = self.center_embed(torch.stack([cx, cy], dim=-1))  # (N, d_model//4)

        # 宽高
        w = bbox[:, 2] - bbox[:, 0]  # (N,)
        h = bbox[:, 3] - bbox[:, 1]  # (N,)
        size_feat = self.size_embed(torch.stack([w, h], dim=-1))  # (N, d_model//4)

        # 2D 位置编码（基于左上角）
        x0_norm = bbox[:, 0]
        y0_norm = bbox[:, 1]
        pos_feat = self.pos_encoding(x0_norm, y0_norm)  # (N, d_model)

        # 类别 embedding
        cat_feat = self.cat_embedding(category)  # (N, d_model//2)

        # 融合所有特征
        combined = torch.cat([bbox_feat, cat_feat, pos_feat[:, :d_model // 2]], dim=-1)
        features = self.fusion(combined)

        return features


if __name__ == "__main__":
    # 简单测试
    N = 5
    num_classes = 8
    d_model = 256

    encoder = ElementFeatureEncoder(num_classes, d_model)

    bbox = torch.rand(N, 4)  # 随机归一化 bbox
    category = torch.randint(0, num_classes, (N,))

    features = encoder(bbox, category)
    print(f"Input: bbox {bbox.shape}, category {category.shape}")
    print(f"Output: features {features.shape}")  # 应该 (N, d_model)
    print(f"Test passed!")
