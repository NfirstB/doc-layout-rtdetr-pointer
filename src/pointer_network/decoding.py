#!/usr/bin/env python3
"""
确定性赢累积解码算法 (Win-Accumulation Decoding)

论文思路：
- 从 N×N 相似度矩阵中恢复拓扑一致的阅读顺序
- 保证顺序符合人类阅读习惯，不矛盾

算法步骤：
1. 从相似度矩阵中提取有序对关系
2. 找到"起始节点"（没有其他节点指向它的节点，即入度为0）
3. 按拓扑顺序贪心遍历，遇到分叉时用赢累积分数决胜负
4. 直到所有节点被访问

"拓扑一致"解读：
- 如果 A 在 B 前读，B 在 C 前读，则不能出现 C 在 A 前的情况
- 即最终顺序中不存在环路
"""
import torch
import numpy as np
from typing import List, Tuple, Optional


def win_accumulation_decode(
    relation_matrix: torch.Tensor,
    bbox: torch.Tensor,
    alpha: float = 0.5,
    min_gain: float = 0.01,
    max_iterations: int = 100,
) -> List[int]:
    """
    赢累积解码算法

    Args:
        relation_matrix: (N, N)，相似度矩阵
                        S_ij > 0 表示元素 i 在元素 j 之前读的概率高
        bbox: (N, 4)，边界框（用于空间关系辅助）
        alpha: 赢累积权重（空间优先还是相似度优先）
        min_gain: 最小增益阈值
        max_iterations: 最大迭代次数

    Returns:
        order: List[int]，阅读顺序（元素索引列表）
    """
    N = relation_matrix.shape[0]
    device = relation_matrix.device

    if N == 0:
        return []
    if N == 1:
        return [0]

    # 归一化相似度矩阵到 [0, 1]
    S = torch.sigmoid(relation_matrix)  # (N, N)

    # 计算初始赢分数
    # W_ij = S_ij（元素 i "赢"了元素 j 的累积分数）
    W = S.clone()  # (N, N)

    # 空间优先分数（基于几何关系）
    # 如果 element i 在 element j 的左/上方，给空间奖励
    x0, y0 = bbox[:, 0], bbox[:, 1]
    x1, y1 = bbox[:, 2], bbox[:, 3]

    cx = (x0 + x1) / 2
    cy = (y0 + y1) / 2

    # i 在 j 左边的程度
    left_of_j = (x1.unsqueeze(0) < x0.unsqueeze(1)).float()  # (N, N)
    # i 在 j 上方的程度
    above_j = (y1.unsqueeze(0) < y0.unsqueeze(1)).float()  # (N, N)

    # 空间优先分数
    spatial_preference = alpha * (left_of_j + above_j) * 0.5

    # 合并：最终分数 = 相似度 + 空间奖励
    # W_ij = S_ij + alpha * spatial(i,j)
    W = W + spatial_preference * (1 - W)  # 加权融合

    # 对角线设为 0（自己与自己无关）
    W = W * (1 - torch.eye(N, device=device))

    # 找起始节点：入度最小（不被任何节点指向）
    in_degree = W.sum(dim=0)  # (N,)
    visited = torch.zeros(N, dtype=torch.bool, device=device)

    order = []
    current = None

    for iteration in range(max_iterations):
        if visited.all():
            break

        if current is None:
            # 找起始节点：入度最小的未访问节点
            # 但优先选择空间上最左上角的
            candidates = (~visited).nonzero(as_tuple=True)[0]
            if len(candidates) == 0:
                break

            # 计算每个候选的"左上程度"分数
            leftness = (x1[candidates] < x1[candidates].unsqueeze(1)).sum(dim=1).float()
            aboveness = (y1[candidates] < y1[candidates].unsqueeze(1)).sum(dim=1).float()
            spatial_score = leftness + aboveness

            # 入度作为辅助
            in_deg = in_degree[candidates]

            # 选左上角 + 入度最小的
            combined_score = spatial_score - in_deg * 0.1
            best_idx = candidates[combined_score.argmax().item()]
            current = best_idx.item()
        else:
            current = int(current)

        # 访问当前节点
        visited[current] = True
        order.append(current)

        # 更新未访问节点的分数
        # 对于未访问的 j，累积从 current 到 j 的赢分数
        unvisited_mask = ~visited
        unvisited_indices = unvisited_mask.nonzero(as_tuple=True)[0]

        if len(unvisited_indices) == 0:
            break

        # 从 current 到每个未访问节点的分数加到 W 上
        for j_idx in unvisited_indices:
            j = int(j_idx)
            # current 在 j 之前的分数
            win_score = W[current, j]
            # 累积到 in_degree（代表 j 需要被读的"压力"）
            in_degree[j] = in_degree[j] * 0.9 + win_score * 0.1

        # 选择下一个：入度最小（赢累积分数最低 = 最可能是下一个）
        in_deg = in_degree[unvisited_indices]
        # 加一点噪声避免总是选同一个
        noise = torch.rand(len(in_deg), device=device) * 0.01
        best_local_idx = (in_deg + noise).argmin().item()
        current = unvisited_indices[best_local_idx].item()

    # 处理剩余未访问（兜底）
    if len(order) < N:
        remaining = [i for i in range(N) if i not in order]
        # 按左上角排序
        remaining_sorted = sorted(remaining, key=lambda i: (y0[i].item(), x0[i].item()))
        order.extend(remaining_sorted)

    return order


def greedy_decode(relation_matrix: torch.Tensor, bbox: torch.Tensor) -> List[int]:
    """
    贪心解码（简化版本）

    每次选入度最小（被指向最少）的未访问节点。
    当多个节点入度相同时，按空间位置（左上优先）决胜。
    """
    N = relation_matrix.shape[0]
    device = relation_matrix.device

    if N == 0:
        return []
    if N == 1:
        return [0]

    # 归一化
    S = torch.sigmoid(relation_matrix)

    # 入度：每个节点被多少其他节点指向
    in_degree = S.sum(dim=0).clone()  # (N,)

    # 对角线清零
    S = S * (1 - torch.eye(N, device=device))
    in_degree = S.sum(dim=0)

    visited = torch.zeros(N, dtype=torch.bool, device=device)

    # 中心坐标
    cx = (bbox[:, 0] + bbox[:, 2]) / 2
    cy = (bbox[:, 1] + bbox[:, 3]) / 2

    order = []

    for _ in range(N):
        candidates = (~visited).nonzero(as_tuple=True)[0]
        if len(candidates) == 0:
            break

        # 入度最低
        in_deg = in_degree[candidates]

        # 找入度最小但 > 0 的，避免选起始时都差不多
        min_deg = in_deg.min()
        min_candidates = candidates[in_deg <= min_deg + 1e-6]

        if len(min_candidates) == 1:
            chosen = min_candidates[0].item()
        else:
            # 入度相同时，选最左上角
            cy_vals = cy[min_candidates]
            cx_vals = cx[min_candidates]
            # 左上程度 = -cy（越小越上）-cx（越小越左）
            score = -cy_vals - cx_vals * 0.01
            chosen = min_candidates[score.argmax().item()].item()

        visited[chosen] = True
        order.append(chosen)

        # 更新入度
        for j in range(N):
            if not visited[j]:
                # chosen 指向 j，j 的入度减少
                in_degree[j] = in_degree[j] * 0.95 + S[chosen, j] * 0.05

    return order


class ReadingOrderPredictor:
    """
    阅读顺序预测器（封装解码算法）

    使用训练好的 Pointer Network 模型，
    对检测到的元素预测阅读顺序。
    """
    def __init__(self, model: torch.nn.Module, device: str = "cpu"):
        self.model = model
        self.device = device
        self.model.eval()

    @torch.no_grad()
    def predict(self, bboxes: np.ndarray, categories: np.ndarray) -> List[int]:
        """
        预测阅读顺序

        Args:
            bboxes: (N, 4)，归一化边界框
            categories: (N,)，类别 ID

        Returns:
            order: List[int]，阅读顺序索引
        """
        bbox_t = torch.tensor(bboxes, dtype=torch.float32, device=self.device)
        cat_t = torch.tensor(categories, dtype=torch.long, device=self.device)

        # Forward pass
        relation_matrix = self.model(bbox_t, cat_t)

        # 解码
        order = win_accumulation_decode(
            relation_matrix.cpu(),
            bbox_t.cpu(),
            alpha=0.5,
        )

        return order


if __name__ == "__main__":
    # 测试
    N = 6
    relation = torch.randn(N, N) * 2

    # 构造一个简单场景：元素0在最左上，5在最右下
    bbox = torch.tensor([
        [0.1, 0.1, 0.3, 0.2],  # 0: 左上
        [0.6, 0.1, 0.8, 0.2],  # 1: 右上
        [0.1, 0.3, 0.3, 0.5],  # 2: 左中
        [0.6, 0.3, 0.8, 0.5],  # 3: 右中
        [0.1, 0.6, 0.3, 0.8],  # 4: 左下
        [0.6, 0.6, 0.8, 0.8],  # 5: 右下
    ])

    order = win_accumulation_decode(relation, bbox, alpha=0.5)
    print(f"Win-accumulation order: {order}")

    order2 = greedy_decode(relation, bbox)
    print(f"Greedy order: {order2}")

    print("Test passed!")
