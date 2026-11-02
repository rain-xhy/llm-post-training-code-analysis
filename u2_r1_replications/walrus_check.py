"""u2：复现 Open-Reasoner-Zero playground/orz_7b_ppo.py:263-265 的 GRPO 归一化写法。
只复刻这三行的语句结构（海象运算符与比较的优先级），不 import 该仓库。
"""
import numpy as np

group = [1.0, 0.0, 1.0, 0.0]            # 同一提示下 4 条回答的 0/1 分数
scores = list(group)
mean = np.mean(group)
for i in range(len(scores)):
    scores[i] -= mean
    if std := np.std(group) > 0:        # 与源码同形：std 取到的是比较结果
        scores[i] /= std
print("std 的类型与取值:", type(std).__name__, std)
print("真实标准差:", float(np.std(group)))
print("实际得到的分数:", scores)
print("按真实标准差归一化应得:", [(g - mean) / float(np.std(group)) for g in group])
