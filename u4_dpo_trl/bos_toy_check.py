"""CPU 玩具验证：用 TRL v0.9.6 自带的 tokenize_row 处理一条偏好样本，
观察 Qwen2.5 分词器 bos_token_id 为 None 时 prompt_input_ids 里是否混入 None。
不构造 DPOTrainer（避免占显存），用最小桩对象提供 tokenize_row 读取的属性。"""
import sys, types
from transformers import AutoTokenizer
from trl import DPOTrainer

MODEL = r"E:\desktop\北大软微\课程\大语言模型应用与实践\作业\13\代码\part2_code_analysis\_models\Qwen2.5-0.5B-Instruct"
tok = AutoTokenizer.from_pretrained(MODEL)
print("bos_token_id:", tok.bos_token_id, "eos_token_id:", tok.eos_token_id)

stub = types.SimpleNamespace(
    is_encoder_decoder=False, is_vision_model=False, tokenizer=tok,
    max_length=256, max_prompt_length=128, truncation_mode="keep_end",
    label_pad_token_id=-100,
)
# build_tokenized_answer 是实例方法，绑定到桩上
stub.build_tokenized_answer = types.MethodType(DPOTrainer.build_tokenized_answer, stub)
feat = {"prompt": "1+1=", "chosen": " 2", "rejected": " 3"}
out = DPOTrainer.tokenize_row(stub, feat)
for k in ["prompt_input_ids", "chosen_input_ids", "rejected_input_ids", "chosen_labels"]:
    print(k, out[k])
print("含 None 的键:", [k for k, v in out.items() if isinstance(v, list) and any(x is None for x in v)])
