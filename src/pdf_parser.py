"""
PDF 解析模块：提取文本块、表格、图片的坐标和内容
"""
import fitz  # pymupdf
from dataclasses import dataclass, field
from typing import List, Optional
import base64
import io
from PIL import Image


@dataclass
class TextBlock:
    """文本块"""
    page: int
    x0: float; y0: float  # 左上角
    x1: float; y1: float  # 右下角
    text: str
    font_size: float = 0.0
    font_name: str = ""
    is_bold: bool = False
    block_type: str = "body"  # title/header/body/footnote/caption

    @property
    def width(self): return self.x1 - self.x0
    @property
    def height(self): return self.y1 - self.y0
    @property
    def center_x(self): return (self.x0 + self.x1) / 2
    @property
    def center_y(self): return (self.y0 + self.y1) / 2
    @property
    def area(self): return self.width * self.height


@dataclass
class ImageBlock:
    """图片块"""
    page: int
    x0: float; y0: float
    x1: float; y1: float
    image_bytes: Optional[bytes] = None
    caption: str = ""
    block_type: str = "figure"

    @property
    def width(self): return self.x1 - self.x0
    @property
    def height(self): return self.y1 - self.y0


@dataclass
class TableBlock:
    """表格块"""
    page: int
    x0: float; y0: float
    x1: float; y1: float
    html: str = ""
    caption: str = ""
    block_type: str = "table"

    @property
    def width(self): return self.x1 - self.x0
    @property
    def height(self): return self.y1 - self.y0


@dataclass
class PageInfo:
    """单页信息"""
    page_num: int
    width: float
    height: float
    texts: List[TextBlock] = field(default_factory=list)
    images: List[ImageBlock] = field(default_factory=list)
    tables: List[TableBlock] = field(default_factory=list)


class PDFParser:
    """解析 PDF，提取布局元素"""

    def __init__(self, pdf_path: str):
        self.pdf_path = pdf_path
        self.doc = fitz.open(pdf_path)
        self._pages: List[PageInfo] = []

    def parse_all(self) -> List[PageInfo]:
        """解析全 PDF"""
        self._pages = []
        for page_num in range(len(self.doc)):
            page = self.doc[page_num]
            page_info = self._parse_page(page, page_num)
            self._pages.append(page_info)
        return self._pages

    def _parse_page(self, page, page_num: int) -> PageInfo:
        """解析单页"""
        rect = page.rect
        page_info = PageInfo(page_num=page_num, width=rect.width, height=rect.height)

        # 提取文本块
        blocks = page.get_text("dict")["blocks"]
        for block in blocks:
            if block.get("type") == 0:  # 文本块
                for line in block.get("lines", []):
                    for span in line.get("spans", []):
                        tb = TextBlock(
                            page=page_num,
                            x0=span["bbox"][0], y0=span["bbox"][1],
                            x1=span["bbox"][2], y1=span["bbox"][3],
                            text=span["text"].strip(),
                            font_size=span.get("size", 0),
                            font_name=span.get("font", ""),
                            is_bold="Bold" in span.get("font", ""),
                        )
                        tb.block_type = self._guess_block_type(tb)
                        page_info.texts.append(tb)
            elif block.get("type") == 1:  # 图片块
                bbox = block.get("bbox", [])
                if bbox:
                    ib = ImageBlock(
                        page=page_num,
                        x0=bbox[0], y0=bbox[1],
                        x1=bbox[2], y1=bbox[3],
                    )
                    # 尝试提取图片
                    try:
                        images = page.get_images(full=True)
                        for img_idx, img in enumerate(images):
                            xref = img[0]
                            base_image = self.doc.extract_image(xref)
                            if base_image:
                                ib.image_bytes = base_image["image"]
                                break
                    except:
                        pass
                    page_info.images.append(ib)

        return page_info

    def _guess_block_type(self, tb: TextBlock) -> str:
        """根据字体大小/位置猜测块类型"""
        text = tb.text.strip()
        if not text:
            return "body"

        # 检查是否是标题（常见关键词）
        title_keywords = ["abstract", "introduction", "related work", "methodology",
                          "experiment", "conclusion", "references", "acknowledgment",
                          "acknowledgements", "figure", "table", "fig.", "equation", "参考文献"]
        text_lower = text.lower()
        for kw in title_keywords:
            if text_lower.startswith(kw):
                return "header"

        # 字体大小判断（相对判断）
        # 如果是页内最大字体，且较短，很可能是标题
        if tb.font_size > 14 and len(text) < 100:
            return "title"
        if tb.font_size > 12 and len(text) < 60 and tb.is_bold:
            return "title"

        # 图片/表格标题
        if text_lower.startswith(("fig", "figure", "table")):
            return "caption"

        # 脚注
        if tb.y1 > self.doc[0].rect.height * 0.9:
            return "footnote"

        return "body"

    def render_page_to_image(self, page_num: int, dpi: int = 150) -> bytes:
        """将页面渲染为图片（用于 YOLO 输入）"""
        page = self.doc[page_num]
        mat = fitz.Matrix(dpi/72, dpi/72)
        pix = page.get_pixmap(matrix=mat)
        return pix.tobytes("png")

    def close(self):
        self.doc.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
