"""只统计模块顶层（含顶层 if/try）定义的名字，函数内的局部导入不算。
静态核对配方文件的 `from verl... import 名字` 在固定 verl 提交里能否解析（不执行 import）。"""
import ast, os, sys
ROOT = sys.argv[1]; FILES = sys.argv[2:]
def mod_path(mod):
    p = os.path.join(ROOT, *mod.split("."))
    return p + ".py" if os.path.isfile(p + ".py") else (os.path.join(p, "__init__.py") if os.path.isfile(os.path.join(p, "__init__.py")) else None)
def defined(path):
    src = open(path, encoding="utf-8").read(); names = set()
    def top(body):
        for n in body:
            yield n
            if isinstance(n, (ast.If, ast.Try)):
                yield from top(n.body); yield from top(getattr(n, "orelse", []))
                for h in getattr(n, "handlers", []): yield from top(h.body)
    for n in top(ast.parse(src).body):
        if isinstance(n, (ast.FunctionDef, ast.ClassDef)): names.add(n.name)
        elif isinstance(n, ast.Assign):
            names.update(t.id for t in n.targets if isinstance(t, ast.Name))
        elif isinstance(n, (ast.Import, ast.ImportFrom)):
            names.update((a.asname or a.name).split(".")[0] for a in n.names)
    return names
for f in FILES:
    print("==", f)
    for n in ast.walk(ast.parse(open(f, encoding="utf-8").read())):
        if isinstance(n, ast.ImportFrom) and n.module and n.module.startswith("verl") and n.level == 0:
            p = mod_path(n.module)
            if p is None: print("  模块不存在:", n.module, [a.name for a in n.names]); continue
            d = defined(p)
            for a in n.names:
                if a.name not in d: print("  名字未找到:", n.module, a.name)
    print("  (检查完毕)")
