"""由日志数值与源码公式推出的几项数量，供代码分析报告引用。
ploo 一侧从固定提交取出原函数文本执行，不重写函数体，其余为公式的直接代入。"""
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
ns = dict(defaultdict=defaultdict, Optional=Optional, np=np, torch=torch, AlgoConfig=object)
exec(compile(ast.fix_missing_locations(ast.Module(body=[fn], type_ignores=[])), SRC, "exec"), ns)
ploo = ns["compute_partition_loo_outcome_advantage"]


def ploo_adv(scores, tau):
    s = torch.tensor(scores, dtype=torch.float32).unsqueeze(-1)
    adv, _ = ploo(s, torch.ones_like(s), np.array(["q"] * len(scores)), tau=tau)
    return adv.squeeze(-1).tolist()


k = 8
print("一、同一组 8 条回答在不同优势估计下的取值，m 为答对条数")
for m in (1, 4, 7):
    r = np.array([1.0] * m + [0.0] * (k - m))
    mean = r.mean()
    g_unb = (r - mean) / (r.std(ddof=1) + 1e-4)      # TRL 0.14.0 GRPOTrainer 的公式
    g_orz_eff = r - mean                              # ORZ 实际效果，只去均值
    g_orz_int = (r - mean) / r.std()                  # ORZ 本意，总体标准差
    print(f"m={m}")
    for name, v in [("GRPO，TRL 公式", g_unb), ("只去均值，ORZ 实际", g_orz_eff), ("除以总体标准差，ORZ 本意", g_orz_int)]:
        print(f"  {name:<22} 答对 {v[0]:+.4f}  答错 {v[-1]:+.4f}  组内和 {v.sum():+.4f}")
    for tau in (1.0, 0.01):
        v = np.array(ploo_adv(r.tolist(), tau))
        print(f"  ploo tau={tau:<5}        答对 {v[0]:+.4f}  答错 {v[-1]:+.4f}  组内和 {v.sum():+.4f}")

print("\n二、ploo 在 tau=0.01 下的闭式与分母下限")
for m in (1, 4, 7):
    if m > 1:
        right = 0.01 * np.log((k - 1) / (m - 1))
    else:
        right = float("nan")
    wrong = -(1 - 0.01 * np.log((k - 1) / m))
    print(f"m={m} 答对闭式 {right:+.4f}  答错闭式 {wrong:+.4f}")
print("m=1 时答对者的其余样本全为 0，分母被下限 1e-8 截住，优势为 -tau*ln(1e-8) =", round(-0.01 * np.log(1e-8), 4), "，未截断时应为 1.0")

print("\n三、DPO 梯度权重 sigma(-margin) 与损失")
sig = lambda x: 1 / (1 + np.exp(-x))
for mg in (0.0, 1.0, 2.0, 3.5256):
    print(f"margin={mg:<7} 权重 {sig(-mg):.4f}  损失 {-np.log(sig(mg)):.4f}")

print("\n四、SimPO 在差值为 0 时的损失 log(1+e^gamma)")
for g in (0.5, 1.0):
    print(f"gamma={g} 损失 {np.log(1 + np.exp(g)):.4f}")
print("DPO 在差值为 0 时的损失", round(float(np.log(2)), 4))

print("\n五、PARL 辅助项，r_parallel 与 r_finish 取日志中的三种编排")
rp = np.array([0.01, 0.2, 1.0])
rf = np.array([1.0, 1.0, 0.1])
for l1, l2 in [(0.1, 0.1), (0.1, 0.11), (0.1, 0.12), (0.1, 0.2)]:
    aux = l1 * rp + l2 * rf
    print(f"lambda1={l1} lambda2={l2}  串行 {aux[0]:.4f}  并行完成 {aux[1]:.4f}  虚假并行 {aux[2]:.4f}  虚假并行低于串行 {aux[2] < aux[0]}")
print("虚假并行低于串行的条件 lambda2 > lambda1 * 0.99 / 0.9 =", round(0.99 / 0.9, 4), "倍 lambda1")
print("初始辅助项之和的上界", 0.1 * 1 + 0.1 * 1)
