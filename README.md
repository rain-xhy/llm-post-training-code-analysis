# 大语言模型后训练算法的代码级分析

本仓库收录一项课程研究的验证脚本与运行日志。研究选取六组论文与开源实现成对阅读，涵盖偏好优化的 InstructGPT、DPO、SimPO，以及面向推理的在线强化学习的 DAPO、DeepSeek-R1、Kimi K2.5，并把论文公式逐条对应到固定提交的代码行。仓库中的脚本用于检验阅读结论，日志是这些脚本和两个最小训练实验的原始输出。

## 目录

| 目录 | 内容 |
|---|---|
| u1_dapo_verl | DAPO 与 veRL，词元级聚合的梯度偏差验证与配方导入核对日志 |
| u2_r1_replications | DeepSeek-R1 的社区复现，Open-Reasoner-Zero 中海象运算符写法的验证 |
| u3_kimi_k25 | Kimi K2.5，PARL 奖励函数与 OpenKimi 优势估计 ploo 的验证 |
| u4_dpo_trl | DPO 与 TRL v0.9.6，偏好对构造、训练脚本、训练日志与曲线 |
| u5_instructgpt_ppo | InstructGPT 的 TRL PPOTrainer 实现，奖励构造、GAE 与裁剪的验证 |
| u6_simpo | SimPO 与 DPO 的并排数值对照 |
| grpo_minimal | 单卡 GRPO 启动实验的脚本与两次运行日志 |
| _shared | 导入检查、推导数值、绘图与论文题录核对等共用脚本 |

每个单元目录下的脚本在同目录的 日志 子目录中有对应输出。

## 验证方法

函数级验证遵循同一原则，即调用框架自身的函数而不重写函数体。涉及 veRL、OpenKimi 与 SimPO 的脚本用抽象语法树从固定提交的源文件中取出目标函数的原文并执行，涉及 TRL 的脚本直接调用已安装的方法，PARL 的脚本按文件路径加载模块。Open-Reasoner-Zero 的验证只复刻三行语句的结构，用于说明 Python 语法行为，并不导入该仓库。

## 被分析的代码版本

脚本默认从与本目录同级的 _repos 目录读取源文件，目录名与固定提交如下。仓库均需自行克隆并检出对应提交，本仓库不包含它们。

| 目录名 | 仓库 | 固定提交 |
|---|---|---|
| verl | github.com/volcengine/verl | a08ea02a784d7d4982b1d81df36d188a476192bb |
| verl-recipe_dapo | github.com/verl-project/verl-recipe 的 dapo 目录 | e7f889574b8301cc0f0fc1d57c6d67f31ffeb689 |
| open-r1 | github.com/huggingface/open-r1 | 5b6ff22b3fb7aa069c54866e517f39dfc3160e09 |
| Open-Reasoner-Zero | github.com/Open-Reasoner-Zero/Open-Reasoner-Zero | 3fdd9a07b4fb01e06005e6e74fc56690cde8a341 |
| PARL | github.com/The-Swarm-Corporation/PARL | 2f984bd1537116529968358bd2babd7f6e403460 |
| OpenKimi | github.com/horizon-rl/OpenKimi | 737515818bf950e0c3f3ee2483ae0c53be435385 |
| trl | github.com/huggingface/trl | 314e8eb367cbfaf74c2e9717085346360e779508 |
| trlx | github.com/CarperAI/trlx | 3340c2f3a56d1d14fdd5f13ad575121fa26b6d92 |
| SimPO | github.com/princeton-nlp/SimPO | 1b3e8f3528a23bce3da514a2dce8ea7490d4bc75 |

## 运行环境

两个最小训练实验在一块 4096 MiB 显存的 RTX 3050 Laptop 上完成，环境为 Python 3.10 与 torch 2.1.2+cu118。DPO 实验使用 TRL 0.9.6、transformers 4.45.2 与 peft 0.13.2，基座为 Qwen2.5-0.5B-Instruct。GRPO 实验使用 TRL 0.14.0 与 transformers 4.46.3，安装在独立目录并通过 PYTHONPATH 置于搜索路径最前。该目录不在仓库中，需要自行安装。

fetch_assets.sh、两处 bos_toy_check.py、grpo_minimal 下的 run.sh 与 train_grpo.py 中写有本地绝对路径，涉及模型目录、Python 解释器位置和 GSM8K 数据文件位置，复现时需要按自己的环境修改。

## 说明

仓库只收录脚本与日志。所有训练实验均为单种子，日志用于说明流程与读码结论，不用于比较算法效果。
