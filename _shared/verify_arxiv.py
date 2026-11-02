"""批量核对 arXiv 编号对应的标题、作者、首次提交日期。

用法: python verify_arxiv.py <编号文件> <输出文件>
编号文件每行一个 arXiv 号，可带 # 注释。输出为制表符分隔文本。
"""
import re
import sys
import time
import urllib.request
import xml.etree.ElementTree as ET

ATOM = "{http://www.w3.org/2005/Atom}"
BATCH = 20


def fetch_batch(ids):
    """请求一批编号，返回 {编号: (标题, 前三作者, 首次提交日期)}。"""
    url = "https://export.arxiv.org/api/query?max_results=%d&id_list=%s" % (len(ids), ",".join(ids))
    with urllib.request.urlopen(url, timeout=60) as resp:
        root = ET.fromstring(resp.read())
    found = {}
    for entry in root.findall(ATOM + "entry"):
        raw_id = entry.findtext(ATOM + "id", "")
        arx = re.sub(r"v\d+$", "", raw_id.rsplit("/abs/", 1)[-1])
        title = " ".join(entry.findtext(ATOM + "title", "").split())
        authors = [a.findtext(ATOM + "name", "") for a in entry.findall(ATOM + "author")]
        published = entry.findtext(ATOM + "published", "")[:10]
        found[arx] = (title, "; ".join(authors[:3]) + (" 等" if len(authors) > 3 else ""), published)
    return found


def main(src, dst):
    with open(src, encoding="utf-8") as f:
        ids = [ln.split("#")[0].strip() for ln in f]
    ids = [i for i in ids if i]
    result = {}
    for start in range(0, len(ids), BATCH):
        chunk = ids[start:start + BATCH]
        try:
            result.update(fetch_batch(chunk))
        except Exception as exc:  # 网络错误如实记录，不静默跳过
            print("batch failed", chunk[0], exc, file=sys.stderr)
        time.sleep(3)  # arXiv API 要求请求间隔
    with open(dst, "w", encoding="utf-8") as out:
        for arx in ids:
            if arx in result:
                title, authors, published = result[arx]
                out.write("\t".join([arx, published, title, authors]) + "\n")
            else:
                out.write("\t".join([arx, "未返回", "", ""]) + "\n")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
