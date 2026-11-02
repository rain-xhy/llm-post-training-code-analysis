"""u6 玩具对照：同一对回答，DPO 与 SimPO 的损失对回答长度的反应。

DPO 一侧调用已安装 TRL 0.9.6 的 DPOTrainer.get_batch_logps / dpo_loss（与克隆的 v0.9.6 文件哈希一致）。
SimPO 一侧用 ast 从固定提交 1b3e8f3 的 scripts/simpo_trainer.py 取 get_batch_logps 与 simpo_loss 的函数文本执行，
不 import 该文件（它依赖 simpo_config 与 TRL 内部符号），也不重写函数体。
用 SimpleNamespace 充当 self，只提供函数体里用到的属性。
"""
import ast, os, textwrap
from types import SimpleNamespace
import torch
import torch.nn.functional as F
from trl import DPOTrainer

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "..", "_repos", "SimPO", "scripts", "simpo_trainer.py")


def extract_method(path, cls, name):
    src = open(path, encoding="utf-8").read()
    for node in ast.parse(src).body:
        if isinstance(node, ast.ClassDef) and node.name == cls:
            for item in node.body:
                if isinstance(item, ast.FunctionDef) and item.name == name:
                    return textwrap.dedent(ast.get_source_segment(src, item, padded=True))
    raise KeyError(name)


ns = {"torch": torch, "F": F, "Tuple": __import__("typing").Tuple}
exec(extract_method(SRC, "SimPOTrainer", "get_batch_logps"), ns)
exec(extract_method(SRC, "SimPOTrainer", "simpo_loss"), ns)
simpo_logps, simpo_loss = ns["get_batch_logps"], ns["simpo_loss"]

torch.manual_seed(0)
V, L = 16, 24
# 两条序列共用同一套 logits 的分布特征：每个位置标签的对数概率大致相同，只是长度不同
logits = torch.randn(2, L, V)
labels = torch.randint(0, V, (2, L))
labels[:, 0] = 5
len_w, len_l = 20, 6                       # chosen 长，rejected 短
labels[0, 1 + len_w:] = -100               # 位置 0 被 shift 丢弃，有效词元 = len
labels[1, 1 + len_l:] = -100

sum_lp, n = DPOTrainer.get_batch_logps(logits, labels)
avg_lp = simpo_logps(logits, labels, average_log_prob=True)
print("有效词元数", n.tolist())
print("DPO 路径(求和)  chosen=%.4f rejected=%.4f 差=%.4f" % (sum_lp[0], sum_lp[1], sum_lp[0] - sum_lp[1]))
print("SimPO 路径(均值) chosen=%.4f rejected=%.4f 差=%.4f" % (avg_lp[0], avg_lp[1], avg_lp[0] - avg_lp[1]))

# 损失：DPO 取参考模型对数概率 = 当前模型（初始化时 policy==ref），此时 logits 项为 0
dev = SimpleNamespace(device="cpu")
dpo_self = SimpleNamespace(accelerator=dev, reference_free=False, f_divergence_type="reverse_kl", f_divergence_params={},
                           loss_type="sigmoid", beta=0.1, label_smoothing=0.0)
for tag, shift in (("policy==ref", 0.0), ("policy 比 ref 把 chosen 提高 1 nat", 1.0)):
    l, cr, rr = DPOTrainer.dpo_loss(dpo_self, sum_lp[:1] + shift, sum_lp[1:], sum_lp[:1], sum_lp[1:])
    print("DPO  loss(%s)=%.4f 隐式奖励差=%.4f" % (tag, l.item(), (cr - rr).item()))

sp_self = SimpleNamespace(accelerator=dev, gamma_beta_ratio=0.5, loss_type="sigmoid", beta=2.0, label_smoothing=0.0)
l, cr, rr = simpo_loss(sp_self, avg_lp[:1], avg_lp[1:])
print("SimPO loss=%.4f 奖励差(β*均值差)=%.4f  γ/β=0.5 β=2.0" % (l.item(), (cr - rr).item()))
# 长度敏感性：只把 chosen 的有效长度从 20 缩到 6，其余不变
labels2 = labels.clone(); labels2[0, 1 + len_l:] = -100
s2, _ = DPOTrainer.get_batch_logps(logits, labels2)
a2 = simpo_logps(logits, labels2, average_log_prob=True)
print("chosen 缩到 6 词元后：DPO 求和 chosen=%.4f (原 %.4f)；SimPO 均值 chosen=%.4f (原 %.4f)" % (s2[0], sum_lp[0], a2[0], avg_lp[0]))
