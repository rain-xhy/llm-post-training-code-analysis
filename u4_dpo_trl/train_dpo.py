"""u4 DPO 最小实验：TRL v0.9.6 DPOTrainer + LoRA，参考模型为关闭适配器的同一模型。

配置见 实验配置记录.md。所有输出写入 日志/ 目录。
"""
import json
import os
import sys
import time

import torch
from datasets import Dataset
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import DPOConfig, DPOTrainer

HERE = os.path.dirname(os.path.abspath(__file__))
LOG = os.path.join(HERE, "日志")
MODEL = os.path.abspath(os.path.join(HERE, "..", "_models", "Qwen2.5-0.5B-Instruct"))


def load_jsonl(path):
    """读取 JSONL，只保留 DPOTrainer 需要的三列。"""
    with open(path, encoding="utf-8") as fh:
        rows = [json.loads(line) for line in fh]
    return Dataset.from_list([{k: r[k] for k in ("prompt", "chosen", "rejected")} for r in rows])


def main():
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    tokenizer.bos_token = "<|endoftext|>"  # 默认 None 会让 tokenize_row 写入 None，见公式对照表末节
    model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.bfloat16).to("cuda")
    peft_config = LoraConfig(r=8, lora_alpha=16, lora_dropout=0.0, task_type="CAUSAL_LM",
                             target_modules=["q_proj", "v_proj"])
    args = DPOConfig(
        output_dir=os.path.join(HERE, "_out"),
        beta=0.1, loss_type="sigmoid", label_smoothing=0.0,
        max_length=96, max_prompt_length=48,
        per_device_train_batch_size=2, gradient_accumulation_steps=4,
        num_train_epochs=3, learning_rate=5e-5, lr_scheduler_type="constant",
        logging_steps=1, eval_strategy="epoch", save_strategy="no",
        bf16=True, report_to="none", seed=0, remove_unused_columns=False,
        precompute_ref_log_probs=False,
    )
    trainer = DPOTrainer(
        model=model, ref_model=None, args=args,
        train_dataset=load_jsonl(os.path.join(LOG, "pairs_train.jsonl")),
        eval_dataset=load_jsonl(os.path.join(LOG, "pairs_eval.jsonl")),
        tokenizer=tokenizer, peft_config=peft_config,
    )
    start = time.time()
    trainer.train()
    elapsed = time.time() - start
    with open(os.path.join(LOG, "log_history.json"), "w", encoding="utf-8") as fh:
        json.dump(trainer.state.log_history, fh, ensure_ascii=False, indent=1)
    peak = torch.cuda.max_memory_allocated() / 2**20
    print(f"[完成] 训练耗时 {elapsed:.1f}s 峰值已分配显存 {peak:.0f} MiB")


if __name__ == "__main__":
    sys.exit(main())
