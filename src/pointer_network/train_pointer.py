#!/usr/bin/env python3
"""
Pointer Network 训练模块

从检测结果和真实阅读顺序标注训练指针网络。

注意：PubLayNet 没有阅读顺序标注。
这里用启发式规则生成伪标签来训练：
- 对于两栏学术论文：按从上到下、从左到右的顺序
- 这个伪标签用于让模型学习基本的空间顺序关系

未来需要真实的阅读顺序标注数据。
"""
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import numpy as np
from typing import List, Tuple, Optional
import json

from .transformer_encoder import PointerNetworkReadingOrder
from .decoding import win_accumulation_decode, greedy_decode


# ── 伪标签生成器 ─────────────────────────────────────────────────────────────

def generate_pseudo_order(bbox: np.ndarray, col_threshold: float = 0.5) -> List[int]:
    """
    生成伪阅读顺序标签

    启发式规则（学术论文）：
    1. 计算两栏分割线位置
    2. 每栏内按 y 坐标排序
    3. 两栏按从上到下交叉合并

    这个函数用于生成训练标签，让模型学习基本的
    "先上后下、先左后右" 顺序。
    """
    N = len(bbox)
    if N == 0:
        return []
    if N == 1:
        return [0]

    x0 = bbox[:, 0]
    y0 = bbox[:, 1]
    x1 = bbox[:, 2]
    y1 = bbox[:, 3]
    cx = (x0 + x1) / 2
    cy = (y0 + y1) / 2

    # 检测是否为两栏布局
    mid_x = (x0.min() + x1.max()) / 2
    left_elements = np.where(cx < mid_x)[0]
    right_elements = np.where(cx >= mid_x)[0]

    # 简单按 x 坐标分组
    if len(left_elements) > 0 and len(right_elements) > 0:
        # 两栏
        left_sorted = sorted(left_elements, key=lambda i: y0[i])
        right_sorted = sorted(right_elements, key=lambda i: y0[i])

        # 交叉合并（Z字形）
        order = []
        max_len = max(len(left_sorted), len(right_sorted))
        for i in range(max_len):
            if i < len(left_sorted):
                order.append(left_sorted[i])
            if i < len(right_sorted):
                order.append(right_sorted[i])
        return order
    else:
        # 单栏，按 y 排序
        return sorted(range(N), key=lambda i: (y0[i], x0[i]))


def compute_order_matrix(order: List[int], N: int) -> np.ndarray:
    """
    将顺序列表转为 N×N 矩阵

    matrix[i,j] = 1 表示 i 在 j 之前
    """
    matrix = np.zeros((N, N), dtype=np.float32)
    for i, elem_i in enumerate(order):
        for j, elem_j in enumerate(order):
            if i < j:
                matrix[elem_i, elem_j] = 1.0
    return matrix


# ── Dataset ─────────────────────────────────────────────────────────────────

class ReadingOrderDataset(Dataset):
    """
    阅读顺序数据集

    输入：检测框 + 类别
    标签：阅读顺序

    由于没有真实标注，用启发式伪标签。
    """
    def __init__(self, data_list: List[Tuple[np.ndarray, np.ndarray]]):
        """
        Args:
            data_list: List of (bboxes, categories) tuples
                      bboxes: (N, 4) 归一化边界框
                      categories: (N,) 类别 ID
        """
        self.data = data_list

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, List[int]]:
        bbox, category = self.data[idx]
        N = len(bbox)

        # 生成伪标签
        order = generate_pseudo_order(bbox)
        order_matrix = compute_order_matrix(order, N)

        return (
            torch.tensor(bbox, dtype=torch.float32),
            torch.tensor(category, dtype=torch.long),
            torch.tensor(order_matrix, dtype=torch.float32),
            order,
        )


# ── 损失函数 ─────────────────────────────────────────────────────────────────

class OrderLoss(nn.Module):
    """
    阅读顺序损失函数

    基于 N×N 相似度矩阵和真实顺序矩阵计算损失。

    正样本（i在j前）：S_ij 应该大
    负样本（i在j后）：S_ij 应该小
    """
    def __init__(self, margin: float = 1.0):
        super().__init__()
        self.margin = margin

    def forward(self, relation_matrix: torch.Tensor,
                order_matrix: torch.Tensor) -> torch.Tensor:
        """
        Args:
            relation_matrix: (N, N)，预测的相似度矩阵
            order_matrix: (N, N)，真实顺序（1=在之前，0=在之后）
        Returns:
            loss: scalar
        """
        N = relation_matrix.shape[0]

        # 正样本对（i在j前）
        positive = order_matrix > 0.5
        # 负样本对
        negative = order_matrix < 0.5

        if positive.sum() == 0:
            return torch.tensor(0.0, device=relation_matrix.device)

        # 正样本：S_ij 应该高
        pos_loss = -relation_matrix[positive].mean()

        # 负样本：S_ij 应该低
        neg_loss = relation_matrix[negative].mean() if negative.sum() > 0 else 0

        # 对角线清零
        diag_mask = 1 - torch.eye(N, device=relation_matrix.device)

        loss = (pos_loss + neg_loss) * diag_mask

        # 对数似然损失（更稳定）
        log_probs = torch.log(torch.sigmoid(relation_matrix) + 1e-6)
        log_probs_neg = torch.log(1 - torch.sigmoid(relation_matrix) + 1e-6)

        # 正样本对的负对数似然
        pos_weight = order_matrix * diag_mask
        neg_weight = (1 - order_matrix) * diag_mask

        loss = -(pos_weight * log_probs).sum() / (pos_weight.sum() + 1e-6) \
               - (neg_weight * log_probs_neg).sum() / (neg_weight.sum() + 1e-6)

        return loss


# ── 训练函数 ─────────────────────────────────────────────────────────────────

def train_pointer_network(
    model: PointerNetworkReadingOrder,
    train_data: List[Tuple[np.ndarray, np.ndarray]],
    val_data: Optional[List[Tuple[np.ndarray, np.ndarray]]] = None,
    num_epochs: int = 30,
    batch_size: int = 32,
    lr: float = 1e-4,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    save_path: str = "pointer_network.pt",
):
    """
    训练指针网络
    """
    model = model.to(device)

    train_dataset = ReadingOrderDataset(train_data)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    if val_data:
        val_dataset = ReadingOrderDataset(val_data)
        val_loader = DataLoader(val_dataset, batch_size=batch_size)
    else:
        val_loader = None

    criterion = OrderLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

    best_val_loss = float('inf')

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        num_batches = 0

        for batch in train_loader:
            bboxes, categories, order_matrices, orders = batch

            #  Padding: 同一 batch 内的 N 可能不同，找最大 N
            max_N = bboxes.shape[1]

            # Mask invalid entries
            valid_mask = torch.arange(max_N, device=device).unsqueeze(0) < bboxes.shape[1]

            optimizer.zero_grad()

            batch_loss = 0.0
            for i in range(bboxes.shape[0]):
                N_i = (bboxes[i, :, 0] != 0).sum().item()
                if N_i == 0:
                    continue

                bbox_i = bboxes[i, :N_i].to(device)
                cat_i = categories[i, :N_i].to(device)
                order_i = order_matrices[i, :N_i, :N_i].to(device)

                relation = model(bbox_i, cat_i)

                loss = criterion(relation, order_i)
                batch_loss += loss

            batch_loss = batch_loss / bboxes.shape[0]
            batch_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            train_loss += batch_loss.item()
            num_batches += 1

        avg_train_loss = train_loss / num_batches if num_batches > 0 else 0
        scheduler.step()

        # 验证
        if val_loader is not None:
            model.eval()
            val_loss = 0.0
            val_batches = 0

            with torch.no_grad():
                for batch in val_loader:
                    bboxes, categories, order_matrices, orders = batch

                    for i in range(bboxes.shape[0]):
                        N_i = (bboxes[i, :, 0] != 0).sum().item()
                        if N_i == 0:
                            continue

                        bbox_i = bboxes[i, :N_i].to(device)
                        cat_i = categories[i, :N_i].to(device)
                        order_i = order_matrices[i, :N_i, :N_i].to(device)

                        relation = model(bbox_i, cat_i)
                        loss = criterion(relation, order_i)
                        val_loss += loss.item()
                        val_batches += 1

            avg_val_loss = val_loss / val_batches if val_batches > 0 else 0

            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                torch.save(model.state_dict(), save_path)

            print(f"Epoch {epoch+1}/{num_epochs} | "
                  f"Train Loss: {avg_train_loss:.4f} | "
                  f"Val Loss: {avg_val_loss:.4f} | "
                  f"LR: {scheduler.get_last_lr()[0]:.6f}")
        else:
            print(f"Epoch {epoch+1}/{num_epochs} | "
                  f"Train Loss: {avg_train_loss:.4f} | "
                  f"LR: {scheduler.get_last_lr()[0]:.6f}")

    return model


# ── 推理接口 ─────────────────────────────────────────────────────────────────

class ReadingOrderModel:
    """
    阅读顺序模型封装

    使用方式：
        model = ReadingOrderModel(
            num_classes=8,
            d_model=256,
            checkpoint_path="pointer_network.pt"
        )
        order = model.predict(bboxes, categories)
    """
    def __init__(
        self,
        num_classes: int = 8,
        d_model: int = 256,
        nhead: int = 8,
        num_layers: int = 6,
        checkpoint_path: Optional[str] = None,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        self.device = device

        self.model = PointerNetworkReadingOrder(
            num_classes=num_classes,
            d_model=d_model,
            nhead=nhead,
            num_layers=num_layers,
        ).to(device)

        if checkpoint_path and os.path.exists(checkpoint_path):
            self.model.load_state_dict(torch.load(checkpoint_path, map_location=device))
            print(f"Loaded checkpoint from {checkpoint_path}")

        self.model.eval()

    @torch.no_grad()
    def predict(self, bboxes: np.ndarray, categories: np.ndarray) -> List[int]:
        """
        预测阅读顺序

        Args:
            bboxes: (N, 4)，归一化边界框
            categories: (N,)，类别 ID

        Returns:
            order: List[int]，阅读顺序
        """
        if len(bboxes) == 0:
            return []
        if len(bboxes) == 1:
            return [0]

        bbox_t = torch.tensor(bboxes, dtype=torch.float32, device=self.device)
        cat_t = torch.tensor(categories, dtype=torch.long, device=self.device)

        relation = self.model(bbox_t, cat_t)

        # 用 win-accumulation 解码
        order = win_accumulation_decode(
            relation.cpu(),
            bbox_t.cpu(),
            alpha=0.5,
        )

        return order

    def save(self, path: str):
        torch.save(self.model.state_dict(), path)

    def compute_accuracy(self, data: List[Tuple[np.ndarray, np.ndarray]]) -> float:
        """
        计算伪标签准确率（用于评估）
        """
        correct = 0
        total = 0

        for bbox, category in data:
            pred_order = self.predict(bbox, category)
            true_order = generate_pseudo_order(bbox)

            # 检查是否匹配
            if pred_order == true_order:
                correct += 1
            total += 1

        return correct / total if total > 0 else 0.0
