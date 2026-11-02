"""生成代码分析报告的图 2 至 图 5。数据全部取自各单元 日志/ 下的原始文件或其中记录的数值。"""
import csv, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
OUT = os.path.abspath(os.path.join(ROOT, "..", "..", "文档", "报告", "图"))
os.makedirs(OUT, exist_ok=True)
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False

# 图 2：u1 微批数与梯度偏差（toy_agg_loss.log）
n = [1, 2, 3, 6]
with_g = [0, 0, 0, 0]
without = [0, 8.08e-2, 1.81e-1, 4.81e-1]
fig, ax = plt.subplots(figsize=(5.6, 3.6))
w = 0.36
x = range(len(n))
ax.bar([i - w / 2 for i in x], with_g, w, label="传入全局词元数")
ax.bar([i + w / 2 for i in x], without, w, label="各微批用本地词元数")
for i, v in enumerate(without):
    ax.text(i + w / 2, v + 0.01, f"{v:.3f}", ha="center", fontsize=8)
    ax.text(i - w / 2, 0.01, "0", ha="center", fontsize=8)
ax.set_xticks(list(x)); ax.set_xticklabels(n)
ax.set_xlabel("微批数"); ax.set_ylabel("梯度最大偏差")
ax.set_title("词元级聚合的梯度偏差（52 个有效词元）"); ax.legend()
fig.tight_layout(); fig.savefig(os.path.join(OUT, "图4_微批梯度偏差.png"), dpi=170); plt.close(fig)

# 图 3：u3 PARL 辅助项退火与三种编排的总奖励
# 分量取自 toy_parl.log：r_parallel=[0.01,0.2,1.0]，r_finish=[1.0,1.0,0.1]，r_perf=0.8；
# 总奖励按 PARLReward.forward 的形式 λ1·r_parallel + λ2·r_finish + r_perf 计算，第 0 与 10000 步与日志一致。
rp = [0.01, 0.2, 1.0]; rf = [1.0, 1.0, 0.1]; perf = 0.8
names = ["1 个子智能体", "20 个且全部完成", "100 个但只完成 10%"]
steps = list(range(0, 12001, 250))
lam = [0.1 * max(0.0, 1 - s / 10000) for s in steps]
fig, ax = plt.subplots(1, 2, figsize=(9.6, 3.6))
ax[0].plot(steps, lam); ax[0].set_title("辅助项系数 λ 的线性退火")
ax[0].set_xlabel("训练步"); ax[0].set_ylabel("λ1 = λ2")
for k in range(3):
    ax[1].plot(steps, [l * rp[k] + l * rf[k] + perf for l in lam], label=names[k])
ax[1].set_title("同一任务结果（r_perf=0.8）下的总奖励")
ax[1].set_xlabel("训练步"); ax[1].set_ylabel("总奖励"); ax[1].legend(fontsize=8)
fig.tight_layout(); fig.savefig(os.path.join(OUT, "图5_PARL奖励退火.png"), dpi=170); plt.close(fig)

# 图 4：GRPO 第二次运行的显存采样（训练窗口 09:17:29–09:20:49）
rows = []
with open(os.path.join(ROOT, "grpo_minimal", "日志", "vram_run2_samples.csv"), encoding="utf-8") as fh:
    for line in fh:
        t, v = line.strip().split(",")
        if "09:17:00" <= t <= "09:21:00":
            h, m, s = map(int, t.split(":"))
            rows.append((h * 3600 + m * 60 + s, int(v)))
t0 = rows[0][0]
fig, ax = plt.subplots(figsize=(7.2, 3.8))
ax.plot([r[0] - t0 for r in rows], [r[1] for r in rows], label="nvidia-smi 读数")
ax.axhline(4096, color="tab:red", ls="--", label="显卡总显存 4096 MiB")
ax.axhline(7502, color="tab:orange", ls=":", label="PyTorch 峰值已分配 7502 MiB")
ax.set_xlabel("自 09:17:00 起的秒数"); ax.set_ylabel("MiB")
ax.set_title("GRPO 启动实验（第二次运行）的显存读数"); ax.legend(fontsize=8)
fig.tight_layout(); fig.savefig(os.path.join(OUT, "图6_GRPO显存读数.png"), dpi=170); plt.close(fig)

# 图 5：u6 求和与均值两条路径对回答长度的反应（toy_compare.log）
fig, ax = plt.subplots(1, 2, figsize=(8.4, 3.6))
for a, title, vals in zip(ax, ["求和路径（DPO）", "均值路径（SimPO）"],
                          [(-69.6325, -23.7882), (-3.4816, -3.9647)]):
    a.bar(["chosen 20 词元", "chosen 缩至 6 词元"], vals, color=["tab:blue", "tab:green"])
    for i, v in enumerate(vals):
        a.text(i, v * 0.94, f"{v:.2f}", ha="center", va="top", fontsize=9, color="white")
    a.set_title(title); a.set_ylabel("chosen 的序列分数")
fig.tight_layout(); fig.savefig(os.path.join(OUT, "图7_求和与均值的长度敏感性.png"), dpi=170); plt.close(fig)
print("done", OUT)
