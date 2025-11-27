
# ---- Proxy setup ----
export http_proxy="http://10.10.78.22:3128"
export https_proxy="$http_proxy"
export HTTP_PROXY="$http_proxy"
export HTTPS_PROXY="$http_proxy"

# Ensure localhost and loopback bypass proxy
export NO_PROXY="127.0.0.1,localhost,$HOSTNAME"
export no_proxy="127.0.0.1,localhost,$HOSTNAME"

# ---- HuggingFace cache dirs (optional, speeds up model loads) ----
export HF_HOME=~/hf
export TRANSFORMERS_CACHE=~/hf/transformers
export HF_DATASETS_CACHE=~/hf/datasets

# ---- Start Worker ----
python -u -m mplug_owl2.serve.model_worker \
    --host 0.0.0.0 \
    --port 31000 \
    --worker-address http://127.0.0.1:31000 \
    --controller-address http://127.0.0.1:7860 \
    --model-path MAGAer13/mplug-owl2-llama2-7b \
    --model-name mplug-owl2-llama2-7b \
    --device cuda \
    --limit-model-concurrency 1 \
    --load-4bit
























