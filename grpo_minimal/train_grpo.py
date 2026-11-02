"""GRPO 最小实验：TRL 0.14.0 GRPOTrainer + LoRA，Qwen2.5-0.5B-Instruct，GSM8K 子集，5 步。

目的只是确认框架能在 4096 MiB 上启动、记录峰值显存与每步耗时，不做效果结论。
奖励是规则函数（答案是否等于 GSM8K 标准答案），没有奖励模型。
"""
import json
import os
import re
import sys
import time

import pandas as pd
import torch
from datasets import Dataset
from peft import LoraConfig
from transformers import AutoTokenizer
from trl import GRPOConfig, GRPOTrainer

HERE = os.path.dirname(os.path.abspath(__file__))
LOG = os.path.join(HERE, "日志")
MODEL = os.path.abspath(os.path.join(HERE, "..", "_models", "Qwen2.5-0.5B-Instruct"))
PARQUET = (r"E:\hf_cache\hub\datasets--openai--gsm8k\snapshots"
           r"\740312add88f781978c0658806c59bc2815b9866\main\train-00000-of-00001.parquet")
N_PROMPTS = int(os.environ.get("N_PROMPTS", "8"))
MAX_STEPS = int(os.environ.get("MAX_STEPS", "5"))
G = int(os.environ.get("NUM_GEN", "8"))
MAX_COMP = int(os.environ.get("MAX_COMP", "256"))

SYSTEM = "Solve the math problem step by step. End your reply with a line of the form '#### <number>'."


def last_number(text):
    """优先取 #### 之后的数，否则取全文最后一个数。"""
    m = re.search(r"####\s*\$?(-?[\d,]*\.?\d+)", text)
    if m:
        s = m.group(1)
    else:
        nums = re.findall(r"-?[\d,]*\.?\d+", text)
        if not nums:
            return None
        s = nums[-1]
    s = s.replace(",", "")
    try:
        return float(s)
    except ValueError:
        return None


def correctness_reward(prompts, completions, answer, **kwargs):
    """完成文本的答案等于标准答案记 1.0，否则 0.0。"""
    out = []
    for comp, gold in zip(completions, answer):
        pred = last_number(comp[0]["content"])
        ref = last_number("#### " + gold.split("####")[-1].strip())
        out.append(1.0 if pred is not None and ref is not None and abs(pred - ref) < 1e-6 else 0.0)
    return out


def main():
    df = pd.read_parquet(PARQUET).head(N_PROMPTS)
    ds = Dataset.from_list([
        {"prompt": [{"role": "system", "content": SYSTEM}, {"role": "user", "content": q}], "answer": a}
        for q, a in zip(df["question"], df["answer"])
    ])
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    args = GRPOConfig(
        output_dir=os.path.join(HERE, "_out"),
        learning_rate=5e-6, lr_scheduler_type="constant",
        per_device_train_batch_size=1, gradient_accumulation_steps=1,
        num_generations=G, max_prompt_length=256, max_completion_length=MAX_COMP,
        temperature=0.9, beta=0.04, max_steps=MAX_STEPS,
        logging_steps=1, save_strategy="no", report_to="none", seed=0,
        bf16=True, remove_unused_columns=False,
        model_init_kwargs={"torch_dtype": torch.bfloat16},
    )
    peft_config = LoraConfig(r=8, lora_alpha=16, lora_dropout=0.0, task_type="CAUSAL_LM",
                             target_modules=["q_proj", "v_proj"])
    trainer = GRPOTrainer(
        model=MODEL, reward_funcs=[correctness_reward], args=args,
        train_dataset=ds, processing_class=tokenizer, peft_config=peft_config,
    )
    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    trainer.train()
    elapsed = time.time() - t0
    with open(os.path.join(LOG, "log_history.json"), "w", encoding="utf-8") as fh:
        json.dump(trainer.state.log_history, fh, ensure_ascii=False, indent=1)
    peak = torch.cuda.max_memory_allocated() / 2**20
    print(f"[完成] 总耗时 {elapsed:.1f}s，{MAX_STEPS} 步，均 {elapsed / MAX_STEPS:.1f}s/步，峰值已分配显存 {peak:.0f} MiB")


if __name__ == "__main__":
    sys.exit(main())
