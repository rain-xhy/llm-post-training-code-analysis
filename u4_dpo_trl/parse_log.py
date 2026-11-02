"""从 日志/log_history.json 解析 DPO 训练指标，写出 metrics.csv 与 metrics_summary.txt。"""
import csv, json, os
HERE = os.path.dirname(os.path.abspath(__file__))
LOG = os.path.join(HERE, "日志")
hist = json.load(open(os.path.join(LOG, "log_history.json"), encoding="utf-8"))
keys = ["step", "epoch", "loss", "rewards/chosen", "rewards/rejected", "rewards/margins", "rewards/accuracies", "learning_rate"]
train = [h for h in hist if "loss" in h and "eval_loss" not in h]
evals = [h for h in hist if "eval_loss" in h]
with open(os.path.join(LOG, "metrics.csv"), "w", newline="", encoding="utf-8") as fh:
    w = csv.writer(fh); w.writerow(keys)
    for h in train: w.writerow([h.get(k) for k in keys])
lines = [f"训练日志条数 {len(train)}，评测条数 {len(evals)}"]
for name, rows in (("train", train[:1] + train[-1:]),):
    for h in rows:
        lines.append(f"{name} step={h['step']} " + " ".join(f"{k}={h.get(k)}" for k in keys[2:7]))
for h in evals:
    lines.append("eval " + " ".join(f"{k}={v}" for k, v in h.items() if k.startswith("eval_rewards") or k in ("eval_loss", "epoch", "step")))
open(os.path.join(LOG, "metrics_summary.txt"), "w", encoding="utf-8").write("\n".join(lines))
print("\n".join(lines))
