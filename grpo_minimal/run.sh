#!/bin/bash
# 用 yolov5-env 的 Python，隔离目录 _site 放在 sys.path 最前面，所有临时与缓存写到 E 盘（C 盘无余量）。
HERE="$(cd "$(dirname "$0")" && pwd)"
mkdir -p /e/tmp_grpo /e/pycache_grpo
export TEMP='E:\tmp_grpo' TMP='E:\tmp_grpo' TMPDIR='E:\tmp_grpo'
export PYTHONPYCACHEPREFIX='E:\pycache_grpo'
export PYTHONPATH="$(cygpath -w "$HERE/_site")"
export HF_HOME='E:\hf_cache' HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 NO_PROXY='*'
export TOKENIZERS_PARALLELISM=false
nvidia-smi --query-gpu=memory.used --format=csv,noheader
/e/anaconda/app/envs/yolov5-env/python.exe -X utf8 "$(cygpath -w "$HERE/train_grpo.py")" 2>&1 | tee "$HERE/日志/train.log"
