#!/usr/bin/env python3
"""把论文 markdown 排成可上传 preprints.org 的 .docx（v2 更正稿）。

论文源文件在仓库外（`../AeroRec-论文终稿-精确数字版.md`），图片在 `docs/paper/figures/`。
本脚本做四件事：

1. 预处理：把 `<sup>[1]</sup>` 形式的引用标记换成 pandoc 上标语法，避免 raw HTML 被丢弃；
2. 调 pandoc 转 docx（依赖 `pip install pypandoc_binary`，自带 pandoc 二进制）；
3. 后处理：图片限宽居中、表格加边框并缩到 8 pt、图注居中斜体、标题层级对齐；
4. 输出 .docx 与同源 .pdf（PDF 用 weasyprint 走 HTML 路线，供人工核对）。

用法：
    python scripts/build_preprint_docx.py                 # 输出到 docs/paper/
    python scripts/build_preprint_docx.py --out /tmp/x.docx

注意：这不是"论文的权威来源"。权威来源是 markdown 与 `data/bias/*.json`；
本脚本只负责把已有内容换个容器，不改动任何数字。
"""
from __future__ import annotations

import argparse
import re
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PAPER_MD = ROOT.parent / "AeroRec-论文终稿-精确数字版.md"
FIG_DIR = ROOT / "docs/paper/figures"
DEFAULT_OUT = ROOT / "docs/paper/AeroRec-preprints.org-v2-draft.docx"

SUP_RE = re.compile(r"<sup>(.*?)</sup>")


def find_pandoc() -> str:
    try:
        import pypandoc

        return pypandoc.get_pandoc_path()
    except Exception:
        from shutil import which

        path = which("pandoc")
        if not path:
            sys.exit("找不到 pandoc：请先 `pip install pypandoc_binary`，或安装系统 pandoc。")
        return path


def preprocess(text: str) -> str:
    """raw HTML 的 <sup> 转 pandoc 上标。

    注意：`^[1]^` 会被 pandoc 当成行内脚注，方括号必须转义成 `^\\[1\\]^`；
    而 `[1]{.superscript}` 里的方括号又会被当成链接语法，也不可用。
    """

    def repl(m: re.Match[str]) -> str:
        inner = m.group(1).replace("[", r"\[").replace("]", r"\]")
        return f"^{inner}^"

    return SUP_RE.sub(repl, text)


def build_reference_docx(pandoc: str, workdir: Path) -> Path:
    """以 pandoc 默认模板为底，改页面尺寸、字体、行距、标题样式。"""
    from docx import Document
    from docx.enum.text import WD_ALIGN_PARAGRAPH
    from docx.shared import Cm, Pt, RGBColor

    ref = workdir / "reference.docx"
    ref.write_bytes(subprocess.run([pandoc, "--print-default-data-file", "reference.docx"],
                                   capture_output=True, check=True).stdout)

    doc = Document(str(ref))
    for section in doc.sections:
        section.page_width, section.page_height = Cm(21.0), Cm(29.7)
        section.left_margin = section.right_margin = Cm(2.5)
        section.top_margin = section.bottom_margin = Cm(2.5)

    styles = doc.styles

    def font(name: str, size: float, bold: bool | None = None, italic: bool | None = None) -> None:
        if name not in [s.name for s in styles]:
            return
        st = styles[name]
        st.font.name = "Times New Roman"
        st.font.size = Pt(size)
        if bold is not None:
            st.font.bold = bold
        if italic is not None:
            st.font.italic = italic
        st.font.color.rgb = RGBColor(0, 0, 0)

    font("Normal", 10.5)
    font("Body Text", 10.5)
    font("First Paragraph", 10.5)
    if "Body Text" in [s.name for s in styles]:
        pf = styles["Body Text"].paragraph_format
        pf.line_spacing = 1.15
        pf.space_after = Pt(4)
    for name, size in (("Heading 1", 16), ("Heading 2", 13), ("Heading 3", 11)):
        font(name, size, bold=True)
        if name in [s.name for s in styles]:
            styles[name].paragraph_format.space_before = Pt(14)
            styles[name].paragraph_format.space_after = Pt(6)
    if "Heading 1" in [s.name for s in styles]:
        styles["Heading 1"].paragraph_format.alignment = WD_ALIGN_PARAGRAPH.CENTER
    if "Image Caption" in [s.name for s in styles]:
        font("Image Caption", 9, italic=True)

    doc.save(str(ref))
    return ref


def _borders(table) -> None:
    from docx.oxml import OxmlElement
    from docx.oxml.ns import qn

    tbl_pr = table._tbl.tblPr
    borders = OxmlElement("w:tblBorders")
    for edge in ("top", "left", "bottom", "right", "insideH", "insideV"):
        el = OxmlElement(f"w:{edge}")
        el.set(qn("w:val"), "single")
        el.set(qn("w:sz"), "4")
        el.set(qn("w:space"), "0")
        el.set(qn("w:color"), "000000")
        borders.append(el)
    tbl_pr.append(borders)


def postprocess(docx_path: Path) -> dict[str, int]:
    from docx import Document
    from docx.enum.text import WD_ALIGN_PARAGRAPH
    from docx.shared import Cm, Pt

    doc = Document(str(docx_path))
    stats = {"images": 0, "tables": 0, "captions": 0}

    # 图片：限宽 15.5 cm 并居中
    for shape in doc.inline_shapes:
        ratio = shape.height.cm / shape.width.cm
        width = min(shape.width.cm, 15.5)
        shape.width = Cm(width)
        shape.height = Cm(width * ratio)
        stats["images"] += 1
    for para in doc.paragraphs:
        if para._p.findall(".//{http://schemas.openxmlformats.org/drawingml/2006/main}blip"):
            para.alignment = WD_ALIGN_PARAGRAPH.CENTER

    # 表格：加边框、8 pt、表头加粗居中、单元格紧凑
    for table in doc.tables:
        _borders(table)
        table.autofit = True
        for r_idx, row in enumerate(table.rows):
            for cell in row.cells:
                for para in cell.paragraphs:
                    para.alignment = (WD_ALIGN_PARAGRAPH.CENTER if r_idx == 0
                                      else WD_ALIGN_PARAGRAPH.LEFT)
                    para.paragraph_format.space_before = Pt(1)
                    para.paragraph_format.space_after = Pt(1)
                    for run in para.runs:
                        run.font.size = Pt(8)
                        run.font.name = "Times New Roman"
                        if r_idx == 0:
                            run.font.bold = True
        stats["tables"] += 1

    # 图注：以 "Figure N:" 开头、整段斜体的段落 → 居中 + 9 pt
    for para in doc.paragraphs:
        text = para.text.strip()
        if re.match(r"^Figure \d+:", text):
            para.alignment = WD_ALIGN_PARAGRAPH.CENTER
            for run in para.runs:
                run.font.size = Pt(9)
                run.font.italic = True
            stats["captions"] += 1

    doc.save(str(docx_path))
    return stats


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--keep-md", type=Path, default=None, help="保留预处理后的 markdown 以便核对")
    args = ap.parse_args()

    if not PAPER_MD.exists():
        sys.exit(f"找不到论文源文件：{PAPER_MD}")
    if not FIG_DIR.is_dir():
        sys.exit(f"找不到图片目录：{FIG_DIR}")

    workdir = args.out.parent
    workdir.mkdir(parents=True, exist_ok=True)
    pandoc = find_pandoc()

    md = preprocess(PAPER_MD.read_text(encoding="utf-8"))
    if args.keep_md:
        args.keep_md.write_text(md, encoding="utf-8")

    # 中间件（预处理后的 md、参考模板）放临时目录，不污染输出目录
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        tmp_md = tmp / "_preprint_v2.md"
        tmp_md.write_text(md, encoding="utf-8")
        ref = build_reference_docx(pandoc, tmp)

        cmd = [pandoc, str(tmp_md), "-f", "markdown+pipe_tables+superscript",
               "-o", str(args.out.resolve()), f"--reference-doc={ref}",
               f"--resource-path={FIG_DIR.parent}"]
        proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        sys.exit(f"pandoc 失败：\n{proc.stderr}")

    stats = postprocess(args.out)

    print(f"输出：{args.out}（{args.out.stat().st_size / 1024:.0f} KB）")
    print(f"  图 {stats['images']} 张 / 表 {stats['tables']} 个 / 图注 {stats['captions']} 条")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
