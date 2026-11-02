"""读取 日志/metrics.csv，画 DPO 损失曲线与奖励曲线，输出 日志/dpo_curves.png。"""
import csv, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
HERE = os.path.dirname(os.path.abspath(__file__))
LOG = os.path.join(HERE, "日志")
rows = list(csv.DictReader(open(os.path.join(LOG, "metrics.csv"), encoding="utf-8")))
step = [int(float(r["step"])) for r in rows]
col = lambda k: [float(r[k]) for r in rows]
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False
fig, ax = plt.subplots(1, 2, figsize=(10, 3.6))
ax[0].plot(step, col("loss")); ax[0].set_title("训练损失"); ax[0].set_xlabel("优化步")
ax[1].plot(step, col("rewards/chosen"), label="被选 chosen")
ax[1].plot(step, col("rewards/rejected"), label="被拒 rejected")
ax[1].plot(step, col("rewards/margins"), label="差值 margin")
ax[1].set_title("隐式奖励 β·log(π/π_ref)"); ax[1].set_xlabel("优化步"); ax[1].legend()
fig.tight_layout(); fig.savefig(os.path.join(LOG, "dpo_curves.png"), dpi=150)
print("saved")
