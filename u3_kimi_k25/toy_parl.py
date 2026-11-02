"""u3 玩具验证：按文件路径加载固定提交的 parl/main.py，调用其 PARLReward 与 CriticalStepsMetric 原方法。
不修改、不重写函数体。验证 λ 退火端点、奖励合成、serial collapse 时 r_parallel 的取值，以及 critical steps 的求法。"""
import importlib.util, os
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
path = os.path.join(HERE, "..", "_repos", "PARL", "parl", "main.py")
spec = importlib.util.spec_from_file_location("parl_main", path)
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)

r = m.PARLReward(lambda1_init=0.1, lambda1_final=0.0, lambda2_init=0.1, lambda2_final=0.0, total_training_steps=10000)
for step in (0, 5000, 10000, 20000):
    print("step=%5d λ1=%.4f λ2=%.4f" % (step, float(r.anneal_lambda1(step)), float(r.anneal_lambda2(step))))

# 同一个任务结果 r_perf，三种编排：串行(1 个子智能体)、并行且完成、虚假并行(100 个但只完成 10%)
num = torch.tensor([1, 20, 100]); comp = torch.tensor([1, 20, 10]); asg = torch.tensor([1, 20, 100])
feat = torch.full((3, 4), 0.8); succ = torch.tensor([1, 1, 1])
for step in (0, 10000):
    out = r.compute_full_reward(num, feat, succ, step, completed_subtasks=comp, assigned_subtasks=asg)
    print("step=%5d total=%s r_parallel=%s r_finish=%s r_perf=%s" % (
        step, [round(x, 4) for x in out["total_reward"].tolist()],
        [round(x, 3) for x in out["r_parallel"].tolist()],
        [round(x, 3) for x in out["r_finish"].tolist()],
        [round(x, 3) for x in out["r_perf"].tolist()]))

# r_perf 的实现：特征均值乘 success；success 为 0 时，不论特征多高都为 0
feat2 = torch.tensor([[0.9, 0.9], [0.9, 0.9]]); succ2 = torch.tensor([1, 0])
print("r_perf(success=1,0) =", r.compute_task_quality(feat2, succ2).tolist())

# critical steps：2 个阶段，阶段 1 有 3 个子智能体步数 [4,9,2]，阶段 2 无子智能体
cs = m.CriticalStepsMetric()
main_steps = torch.tensor([[1.0, 1.0]])
sub = torch.tensor([[[4.0, 9.0, 2.0], [0.0, 0.0, 0.0]]])
print("critical steps =", cs(main_steps, sub).tolist(), "(论文式: (1+9)+(1+0)=11)")
