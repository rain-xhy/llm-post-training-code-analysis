"""u5 玩具验证：调用已安装 TRL 0.9.6 的 PPOTrainer 原方法（compute_rewards、compute_advantages、loss），
用 SimpleNamespace 充当 self，不构造训练器，不重写函数体。
验证三件事：KL 惩罚并入逐词元奖励且分数只加在最后一个有效词元上；GAE 在 gamma=1、lam=1 时等于回报减价值；
PPO 裁剪在 ratio 越界且优势为正时把梯度截断。
"""
import torch
from types import SimpleNamespace
from trl import PPOTrainer

torch.manual_seed(0)
cfg = SimpleNamespace(kl_penalty="kl", gamma=1.0, lam=1.0, whiten_rewards=False,
                      cliprange=0.2, cliprange_value=0.2, vf_coef=0.1, ratio_threshold=10.0)
me = SimpleNamespace(config=cfg, kl_ctl=SimpleNamespace(value=0.2))
me._kl_penalty = lambda lp, rlp: PPOTrainer._kl_penalty(me, lp, rlp)

# 两条回答，有效长度 3 与 2，补到 4
mask = torch.tensor([[1, 1, 1, 0], [1, 1, 0, 0]])
logp = torch.tensor([[-1.0, -2.0, -1.5, 0.0], [-0.5, -1.0, 0.0, 0.0]])
ref = torch.tensor([[-1.2, -1.8, -1.0, 0.0], [-0.5, -0.5, 0.0, 0.0]])
score = torch.tensor([2.0, -1.0])
r, nsr, kl = PPOTrainer.compute_rewards(me, score, logp, ref, mask)
print("KL 项 -beta*(logp-ref):\n", nsr)
print("并入分数后的逐词元奖励:\n", r)
print("分数加在最后有效词元:", [float(r[i, int(mask[i].nonzero()[-1])] - nsr[i, int(mask[i].nonzero()[-1])]) for i in range(2)])

values = torch.tensor([[0.5, 0.4, 0.3, 0.0], [0.2, 0.1, 0.0, 0.0]])
_, adv, ret = PPOTrainer.compute_advantages(me, values, r, mask)
raw = (ret - values * mask)
print("returns:\n", ret)
print("returns - values(未白化优势):\n", raw)
print("白化后优势均值(有效位置):", float((adv * mask).sum() / mask.sum()))

# 裁剪：ratio=1.5，优势为正，cliprange=0.2
old = torch.zeros(1, 2); new = torch.log(torch.tensor([[1.5, 1.0]])).requires_grad_(True)
m = torch.ones(1, 2); a = torch.tensor([[1.0, 1.0]]); v = torch.zeros(1, 2)
logits = torch.randn(1, 2, 5)
pg, vf, _ = PPOTrainer.loss(me, old, v, logits, v.clone(), new, m, a, v.clone())
pg.backward()
print("pg_loss=%.4f 梯度=%s" % (pg.item(), new.grad.tolist()))
