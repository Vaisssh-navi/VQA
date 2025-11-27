# Important: bypass proxy for localhost and loopback
export NO_PROXY="127.0.0.1,localhost,$HOSTNAME"
export no_proxy="127.0.0.1,localhost,$HOSTNAME"

# ---- Launch Gradio Web UI ----
python -u -m mplug_owl2.serve.gradio_web_server \
    --controller-url http://127.0.0.1:7860 \
    --host 0.0.0.0 \
    --port 8080
