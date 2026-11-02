"""构造 DPO 最小实验用的算术偏好对。

chosen 是正确的和，rejected 是偏离正确值 1 到 9 的错误和。
输出 JSONL，每行含 prompt、chosen、rejected。固定随机种子，可重放。
"""
import json
import random
import sys


def build_pairs(count: int, rng: random.Random):
    """生成 count 条偏好对，避免训练集内出现重复的问句。"""
    seen = set()
    pairs = []
    while len(pairs) < count:
        a, b = rng.randint(10, 99), rng.randint(10, 99)
        if (a, b) in seen:
            continue
        seen.add((a, b))
        truth = a + b
        wrong = truth + rng.choice([-1, 1]) * rng.randint(1, 9)
        pairs.append({
            "prompt": f"What is {a} + {b}? Answer:",
            "chosen": f" {truth}",
            "rejected": f" {wrong}",
            "key": [a, b],
        })
    return pairs, seen


def main(out_train: str, out_eval: str, seed: int = 0):
    rng = random.Random(seed)
    train, seen = build_pairs(200, rng)
    # 验证集与训练集的题目互不重叠
    eval_rng = random.Random(seed + 1)
    eval_pairs = []
    while len(eval_pairs) < 20:
        cand, _ = build_pairs(1, eval_rng)
        if tuple(cand[0]["key"]) not in seen:
            eval_pairs.append(cand[0])
            seen.add(tuple(cand[0]["key"]))
    for path, rows in ((out_train, train), (out_eval, eval_pairs)):
        with open(path, "w", encoding="utf-8") as fh:
            for row in rows:
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
