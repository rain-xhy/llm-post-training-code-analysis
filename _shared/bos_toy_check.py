"""CPU 玩具验证：Qwen 分词器 bos_token_id 为 None 时，TRL v0.9.6 的 tokenize_row 输出是否含 None。
不加载模型，不占 GPU。用 SimpleNamespace 充当 self，只提供 tokenize_row 用到的属性。"""
import types
from transformers import AutoTokenizer
from trl import DPOTrainer

MODEL_DIR = r"E:\desktop\北大软微\课程\大语言模型应用与实践\作业\13\代码\part2_code_analysis\_models\Qwen2.5-0.5B-Instruct"
tok = AutoTokenizer.from_pretrained(MODEL_DIR)
print("bos_token_id:", tok.bos_token_id, "eos_token_id:", tok.eos_token_id)

fake_self = types.SimpleNamespace(
    is_encoder_decoder=False, is_vision_model=False, tokenizer=tok,
    max_length=64, max_prompt_length=32, truncation_mode="keep_end",
    label_pad_token_id=-100,
    build_tokenized_answer=None,
)
fake_self.build_tokenized_answer = lambda p, a, images=None: DPOTrainer.build_tokenized_answer(fake_self, p, a, images)
feat = {"prompt": "1+1=", "chosen": " 2", "rejected": " 3"}
out = DPOTrainer.tokenize_row(fake_self, feat)
for k in ["prompt_input_ids", "chosen_input_ids", "rejected_input_ids", "chosen_labels"]:
    print(k, out[k])
print("含 None 的键:", [k for k, v in out.items() if isinstance(v, list) and None in v])
