"""
阅读顺序推理模块
基于 2D 坐标 + 栏位分析 + 空间图模型
"""
from dataclasses import dataclass, field
from typing import List, Tuple, Set, Dict
import numpy as np


@dataclass
class LayoutElement:
    """任意布局元素（文本块/图片/表格）"""
    id: str
    elem_type: str  # title/header/body/figure/table/caption/footnote/equation
    page: int
    x0: float; y0: float
    x1: float; y1: float
    text: str = ""  # 可选，用于语义分析
    confidence: float = 1.0

    @property
    def center_x(self): return (self.x0 + self.x1) / 2
    @property
    def center_y(self): return (self.y0 + self.y1) / 2
    @property
    def width(self): return self.x1 - self.x0
    @property
    def height(self): return self.y1 - self.y0


class ReadingOrderInferencer:
    """
    阅读顺序推理：
    1. 检测栏数（单栏/双栏/三栏）
    2. 按行分组
    3. 行内按 x 排序
    4. 行间按 y 排序
    5. 处理跨栏元素（标题/图片）
    """

    def __init__(self, page_width: float, page_height: float):
        self.page_width = page_width
        self.page_height = page_height
        self.col_count = 1
        self.col_boundaries: List[float] = []

    def detect_columns(self, elements: List[LayoutElement]) -> int:
        """检测栏数：分析页面宽度利用情况"""
        if not elements:
            return 1

        # 计算所有元素的左右边界
        lefts = [e.x0 for e in elements]
        rights = [e.x1 for e in elements]

        # 检查是否有明显的栏分隔
        # 方法：分析元素左边界分布
        margin_left = min(lefts)
        margin_right = self.page_width - max(rights)

        # 内容宽度
        content_width = max(rights) - min(lefts)
        page_content_ratio = content_width / self.page_width

        if page_content_ratio > 0.75:
            # 内容占据大部分页面宽度 → 单栏
            self.col_count = 1
            self.col_boundaries = [0, self.page_width]
        else:
            # 可能双栏
            # 分析左边界是否有双峰分布
            lefts_sorted = sorted(lefts)
            if len(lefts_sorted) > 5:
                gaps = [lefts_sorted[i+1] - lefts_sorted[i] for i in range(len(lefts_sorted)-1)]
                max_gap_idx = np.argmax(gaps)
                gap_ratio = gaps[max_gap_idx] / content_width
                if gap_ratio > 0.15:
                    # 双栏
                    self.col_count = 2
                    mid = (lefts_sorted[max_gap_idx] + lefts_sorted[max_gap_idx+1]) / 2
                    self.col_boundaries = [0, mid, self.page_width]
                else:
                    self.col_count = 1
                    self.col_boundaries = [0, self.page_width]

        return self.col_count

    def infer(self, elements: List[LayoutElement]) -> List[LayoutElement]:
        """
        推理阅读顺序
        返回：按阅读顺序排列的元素列表
        """
        if not elements:
            return []

        # 检测栏数
        self.detect_columns(elements)

        # 分离跨栏元素（标题、图片、表格）和普通文本块
        spanning, normal = self._separate_spanning(elements)

        # 对普通元素排序
        sorted_normal = self._sort_elements(normal)

        # 将跨栏元素插入合适位置
        result = self._insert_spanning(sorted_normal, spanning)

        return result

    def _separate_spanning(self, elements: List[LayoutElement]) -> Tuple[List[LayoutElement], List[LayoutElement]]:
        """分离跨栏元素和普通元素"""
        spanning = []
        normal = []

        # 跨栏判断：元素宽度 > 单栏宽度的 80%
        col_width = self.page_width / self.col_count if self.col_count > 0 else self.page_width

        for e in elements:
            if e.elem_type in ("title", "header", "figure", "table"):
                if e.width > col_width * 0.8:
                    spanning.append(e)
                else:
                    normal.append(e)
            else:
                normal.append(e)

        return spanning, normal

    def _sort_elements(self, elements: List[LayoutElement]) -> List[LayoutElement]:
        """核心排序：先按行分组，再按列排序"""
        if not elements:
            return []

        # 估算平均行高（作为行间距参考）
        ys = [(e.y0 + e.y1) / 2 for e in elements]
        if len(ys) < 2:
            return sorted(elements, key=lambda e: (e.page, e.center_y, e.center_x))

        # 用聚类检测行
        row_height = self._estimate_row_height(elements)
        if row_height < 5:
            row_height = 20  # 默认行高

        # 给每个元素分配行号
        for e in elements:
            e._row = int(round(e.center_y / row_height))

        # 按 (page, row, col) 排序
        col_width = self.page_width / self.col_count if self.col_count > 0 else self.page_width
        for e in elements:
            e._col = int((e.center_x / col_width) if self.col_count > 1 else 0)

        sorted_elems = sorted(elements, key=lambda e: (e.page, e._row, e._col))

        # 清理临时属性
        for e in elements:
            if hasattr(e, '_row'): delattr(e, '_row')
            if hasattr(e, '_col'): delattr(e, '_col')

        return sorted_elems

    def _estimate_row_height(self, elements: List[LayoutElement]) -> float:
        """估算平均行高（基于 y 坐标分布）"""
        if len(elements) < 3:
            return 20.0

        centers_y = sorted([(e.y0 + e.y1) / 2 for e in elements])
        # 计算相邻中心点的间距
        gaps = [centers_y[i+1] - centers_y[i] for i in range(len(centers_y)-1)]
        # 取中位数
        gaps_sorted = sorted(gaps)
        if len(gaps_sorted) % 2 == 0:
            median_gap = (gaps_sorted[len(gaps_sorted)//2 - 1] + gaps_sorted[len(gaps_sorted)//2]) / 2
        else:
            median_gap = gaps_sorted[len(gaps_sorted)//2]

        return max(median_gap, 10)

    def _insert_spanning(self, sorted_normal: List[LayoutElement], spanning: List[LayoutElement]) -> List[LayoutElement]:
        """将跨栏元素（标题、图片）插入合适位置"""
        if not spanning:
            return sorted_normal

        if not sorted_normal:
            return sorted(spanning, key=lambda e: (e.page, e.center_y, e.center_x))

        result = []
        spanning_idx = 0

        for e in sorted_normal:
            # 找到当前元素之前最近的跨栏元素
            while spanning_idx < len(spanning) and \
                  (spanning[spanning_idx].page < e.page or
                   (spanning[spanning_idx].page == e.page and
                    spanning[spanning_idx].y1 <= e.y0 + 5)):
                result.append(spanning[spanning_idx])
                spanning_idx += 1
            result.append(e)

        # 剩余的跨栏元素放最后
        while spanning_idx < len(spanning):
            result.append(spanning[spanning_idx])
            spanning_idx += 1

        return result


def elements_to_json(elements: List[LayoutElement]) -> List[Dict]:
    """转换为 JSON 友好格式"""
    return [
        {
            "id": e.id,
            "type": e.elem_type,
            "page": e.page,
            "bbox": [round(e.x0, 1), round(e.y0, 1), round(e.x1, 1), round(e.y1, 1)],
            "text": e.text[:200] if e.text else "",
            "reading_order": i,
        }
        for i, e in enumerate(elements)
    ]
