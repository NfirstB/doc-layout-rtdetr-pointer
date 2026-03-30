"""
Pointer Network 模块

论文架构实现：
- 2D绝对位置编码 + 类别embedding
- 6层Transformer编码器 + 几何偏置（Relation-DETR风格）
- 成对关系头（N×N相似度矩阵）
- 赢累积解码算法（确定性拓扑一致的阅读顺序）

使用方式：
    from pointer_network import (
        PointerNetworkReadingOrder,
        ReadingOrderModel,
        train_pointer_network,
        win_accumulation_decode,
    )
"""
from .position_encoding import (
    Sinusoidal2DPositionalEncoding,
    CategoryEmbedding,
    ElementFeatureEncoder,
)
from .transformer_encoder import (
    GeometricBias,
    TransformerEncoderLayer,
    TransformerEncoder,
    PairwiseRelationHead,
    PointerNetworkReadingOrder,
)
from .decoding import (
    win_accumulation_decode,
    greedy_decode,
    ReadingOrderPredictor,
)
from .train_pointer import (
    ReadingOrderModel,
    train_pointer_network,
    generate_pseudo_order,
    compute_order_matrix,
    OrderLoss,
    ReadingOrderDataset,
)

__all__ = [
    # Position encoding
    "Sinusoidal2DPositionalEncoding",
    "CategoryEmbedding",
    "ElementFeatureEncoder",
    # Transformer
    "GeometricBias",
    "TransformerEncoderLayer",
    "TransformerEncoder",
    "PairwiseRelationHead",
    "PointerNetworkReadingOrder",
    # Decoding
    "win_accumulation_decode",
    "greedy_decode",
    "ReadingOrderPredictor",
    # Training
    "ReadingOrderModel",
    "train_pointer_network",
    "generate_pseudo_order",
    "compute_order_matrix",
    "OrderLoss",
    "ReadingOrderDataset",
]
