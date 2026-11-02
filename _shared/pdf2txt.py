import sys, glob, os
import pymupdf
for d in sys.argv[1:]:
    for p in sorted(glob.glob(os.path.join(d, "*.pdf"))):
        t = p[:-4] + ".txt"
        if os.path.exists(t): continue
        doc = pymupdf.open(p)
        open(t, "w", encoding="utf-8").write("\n".join(pg.get_text() for pg in doc))
        print(os.path.basename(p), len(doc))
