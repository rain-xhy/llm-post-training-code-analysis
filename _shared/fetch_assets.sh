#!/usr/bin/env bash
# 后台下载第二部分需要的外部资产：0.5B 模型权重 + 各仓库浅克隆。
# 每个仓库放在 _repos 下各自独立的目录，脚本放在 _shared，不在仓库目录里运行任何代码。
# 每个资产的结果（成功/失败、提交号、提交日期）追加写入 fetch_assets.log，失败不中断后续。

ROOT="/e/desktop/北大软微/课程/大语言模型应用与实践/作业/13/代码/part2_code_analysis"
REPOS="$ROOT/_repos"
LOG="$ROOT/_shared/fetch_assets.log"
PY="/e/anaconda/app/envs/yolov5-env/python"

mkdir -p "$REPOS" "$ROOT/_models"
export NO_PROXY='*'
export HF_ENDPOINT=https://hf-mirror.com

echo "=== 开始 $(date '+%F %T') ===" >> "$LOG"

# name|url|ref(空表示默认分支)
REPO_LIST=(
  "trl|https://github.com/huggingface/trl.git|v0.9.6"
  "verl|https://github.com/volcengine/verl.git|"
  "SimPO|https://github.com/princeton-nlp/SimPO.git|"
  "trlx|https://github.com/CarperAI/trlx.git|"
  "open-r1|https://github.com/huggingface/open-r1.git|"
  "Open-Reasoner-Zero|https://github.com/Open-Reasoner-Zero/Open-Reasoner-Zero.git|"
  "PARL|https://github.com/The-Swarm-Corporation/PARL.git|"
  "OpenKimi|https://github.com/horizon-rl/OpenKimi.git|"
)

for entry in "${REPO_LIST[@]}"; do
  IFS='|' read -r name url ref <<< "$entry"
  dest="$REPOS/$name"
  if [ -d "$dest/.git" ]; then
    echo "[跳过] $name 已存在" >> "$LOG"
    continue
  fi
  if [ -n "$ref" ]; then
    timeout 600 git clone --depth 1 --branch "$ref" "$url" "$dest" >> "$LOG" 2>&1
  else
    timeout 600 git clone --depth 1 "$url" "$dest" >> "$LOG" 2>&1
  fi
  if [ $? -eq 0 ]; then
    commit=$(git -C "$dest" rev-parse HEAD)
    cdate=$(git -C "$dest" log -1 --format=%cI)
    echo "[成功] $name url=$url ref=${ref:-默认分支} commit=$commit commit_date=$cdate fetched=$(date '+%F')" >> "$LOG"
  else
    echo "[失败] $name url=$url" >> "$LOG"
    rm -rf "$dest"
  fi
done

# 模型权重：只取推理与训练需要的文件
"$PY" - >> "$LOG" 2>&1 <<'EOF'
from huggingface_hub import snapshot_download
import os
root = os.environ.get("MODEL_ROOT", "")
path = snapshot_download(
    repo_id="Qwen/Qwen2.5-0.5B-Instruct",
    local_dir=r"E:\desktop\北大软微\课程\大语言模型应用与实践\作业\13\代码\part2_code_analysis\_models\Qwen2.5-0.5B-Instruct",
    allow_patterns=["*.json", "*.safetensors", "merges.txt", "vocab.json", "tokenizer*"],
)
print("[成功] 模型 Qwen2.5-0.5B-Instruct ->", path)
EOF
[ $? -ne 0 ] && echo "[失败] 模型下载" >> "$LOG"

echo "=== 结束 $(date '+%F %T') ===" >> "$LOG"
