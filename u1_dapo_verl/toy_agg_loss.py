"""u1 核对点二的玩具验证。

从固定提交的 verl/trainer/ppo/core_algos.py 里用 ast 取出 agg_loss 的原函数文本并执行，
不重写函数体。verl 顶层包导入需要 ray，本机环境没有，所以不 import verl。
masked_sum 来自 verl/utils/torch_functional.py，同样按源码文本取出。
验证内容：同一个 mini-batch 切成不同微批数，token-mean 下累积梯度是否等于全局词元平均。
"""
import ast
import os
import sys
import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "_repos", "verl", "verl"))


def extract_func(path, name):
    """返回源文件中名为 name 的顶层函数的源码文本。"""
    src = open(path, encoding="utf-8").read()
    for node in ast.parse(src).body:
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return ast.get_source_segment(src, node)
    raise KeyError(name)


ns = {"torch": torch, "Optional": __import__("typing").Optional}
tf = os.path.join(ROOT, "utils", "torch_functional.py")
exec(extract_func(tf, "masked_sum"), ns)


class _VF:  # 只提供 agg_loss 用到的 masked_sum
    masked_sum = staticmethod(ns["masked_sum"])


ns["verl_F"] = _VF
exec(extract_func(os.path.join(ROOT, "trainer", "ppo", "core_algos.py"), "agg_loss"), ns)
agg_loss = ns["agg_loss"]

torch.manual_seed(0)
# 一个 mini-batch：6 条回答，长度差异大，逐词元损失随机
lens = [2, 3, 5, 8, 13, 21]
T = max(lens)
loss_mat = torch.randn(6, T, requires_grad=True)
mask = torch.zeros(6, T)
for i, n in enumerate(lens):
    mask[i, :n] = 1
total_tokens = int(mask.sum())
print("总词元数", total_tokens)

# 基准：全局词元平均
ref = (loss_mat * mask).sum() / total_tokens
ref.backward()
g_ref = loss_mat.grad.clone()
loss_mat.grad = None

def run(splits, use_global):
    """splits 为各微批的样本下标；use_global 决定是否传入全局词元数。"""
    loss_mat.grad = None
    acc = 0.0
    for idx in splits:
        kw = {"batch_num_tokens": total_tokens} if use_global else {}
        l = agg_loss(loss_mat[idx], mask[idx], "token-mean", dp_size=1, **kw)
        l.backward()
        acc += float(l)
    return acc, loss_mat.grad.clone()

for name, splits in {"1个微批": [[0,1,2,3,4,5]], "2个微批": [[0,1,2],[3,4,5]],
                     "3个微批": [[0,1],[2,3],[4,5]], "6个微批": [[i] for i in range(6)]}.items():
    for use_global in (True, False):
        s, g = run(splits, use_global)
        diff = (g - g_ref).abs().max().item()
        print(f"{name} 传入全局词元数={use_global} 损失和={s:.6f} 基准={float(ref):.6f} 梯度最大偏差={diff:.2e}")
