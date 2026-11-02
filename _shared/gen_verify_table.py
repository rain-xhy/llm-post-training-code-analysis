"""由 arxiv_check.tsv 生成文献核实表的表格主体。

输入 tsv 的列为：编号、v1 提交日期、标题、作者（前三位加"等"）。
核实状态只有两档来源，一是交接说明里记录的已读原文范围，二是 arXiv API 返回的题录。
没有读过正文的条目一律标为"仅核题录"。
"""
import sys
from pathlib import Path

# 已读原文的范围，取自《第一二部分_交接说明.md》第四节，不在此处扩写。
READ_SCOPE = {
    "2502.04463": "已读原文",
    "2501.12599": "已读原文",
    "2602.02276": "已读原文（Toggle 一节）",
    "2505.15612": "已读原文",
    "2506.05256": "已读原文",
    "2505.18822": "已读部分原文",
    "2510.15110": "读摘要、引言、实验片段",
    "2602.09591": "读摘要、引言、实验设置",
    "2605.00365": "读摘要、引言、实验设置",
}
DEFAULT_STATUS = "仅核题录，未读正文"


def load_rows(tsv_path: Path):
    """读取 tsv，返回 (编号, 日期, 标题, 作者) 列表，跳过空行。"""
    rows = []
    for line in tsv_path.read_text(encoding="utf-8").splitlines():
        parts = line.split("\t")
        if len(parts) < 4:
            continue
        rows.append(tuple(p.strip() for p in parts[:4]))
    return rows


def render(rows) -> str:
    """把行渲染成 markdown 表格，标题中的竖线转义以免破坏表格。"""
    lines = ["| arXiv 编号 | 提交日期 | 标题 | 作者 | 核实状态 |", "|---|---|---|---|---|"]
    for arxiv_id, date, title, authors in rows:
        status = READ_SCOPE.get(arxiv_id, DEFAULT_STATUS)
        lines.append(
            f"| {arxiv_id} | {date} | {title.replace('|', '/')} | {authors} | {status} |"
        )
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit("用法 gen_verify_table.py <输入tsv> <输出md>")
    rows = load_rows(Path(sys.argv[1]))
    Path(sys.argv[2]).write_text(render(rows), encoding="utf-8")
    print(f"{len(rows)} 行，已写入 {sys.argv[2]}")
