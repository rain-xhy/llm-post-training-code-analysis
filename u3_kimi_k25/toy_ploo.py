"""u3 玩具验证：从固定提交的 core_algos.py 取出 ploo 优势估计函数文本执行，不重写函数体。
观察 tau=0.01 时留一基线在不同组成的一组回答上给出的优势，重点是分母下限截断 clamp(min=1e-8) 的影响。"""
import ast
import os
from collections import defaultdict
from typing import Optional

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "..", "_repos", "OpenKimi", "openkimi", "backends", "verl", "core_algos.py")
tree = ast.parse(open(SRC, encoding="utf-8").read())
fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "compute_partition_loo_outcome_advantage")
fn.decorator_list = []
mod = ast.Module(body=[fn], type_ignores=[])
ns = dict(defaultdict=defaultdict, Optional=Optional, np=np, torch=torch, AlgoConfig=object)
exec(compile(ast.fix_missing_locations(mod), SRC, "exec"), ns)
ploo = ns["compute_partition_loo_outcome_advantage"]


def run(scores, tau=0.01):
    s = torch.tensor(scores, dtype=torch.float32).unsqueeze(-1)
    mask = torch.ones_like(s)
    idx = np.array(["q"] * len(scores))
    adv, _ = ploo(s, mask, idx, tau=tau)
    return [round(x, 4) for x in adv.squeeze(-1).tolist()]


cases = {
    "8 条中 1 条正确": [1, 0, 0, 0, 0, 0, 0, 0],
    "8 条中 4 条正确": [1, 1, 1, 1, 0, 0, 0, 0],
    "8 条中 7 条正确": [1, 1, 1, 1, 1, 1, 1, 0],
    "8 条全对": [1] * 8,
    "8 条全错": [0] * 8,
}
for tau in (0.01, 1.0):
    print("tau =", tau)
    for k, v in cases.items():
        print("  %-12s %s" % (k, run(v, tau)))
print("log(1e-8) * 0.01 =", round(float(np.log(1e-8)) * 0.01, 4))
