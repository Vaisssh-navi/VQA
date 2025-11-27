# 📁 Project Structure & Usage

This repository contains the complete pipeline for running, benchmarking, and fine‑tuning a Chart‑VQA system based on mPLUG‑Owl2.

---

## 🖥️ UI Folder  
Directory: `UI/`

Shell scripts to launch the full Controller–Worker–UI system.

| File | Description |
|------|-------------|
| `controller.sh` | Starts the Controller server |
| `worker.sh` | Starts the Worker (loads mPLUG‑Owl2 model) |
| `web.sh` | Starts the Gradio Web UI |

### **Run the full pipeline**
```bash
bash UI/controller.sh
bash UI/worker.sh
bash UI/web.sh
```

---

## 📊 Benchmark Folder  
Directory: `benchmark/`

Scripts to run and evaluate the MMC Benchmark datasets.

| File | Description |
|------|-------------|
| `run_mmc_mqa.py` | Runs inference on MMC‑MQA (image + question) |
| `eval_mmc_mqa.py` | Evaluates predictions for MMC‑MQA |
| `run_mmc_text_full2.py` | Runs inference on MMC‑Text (text‑only reasoning) |
| `eval_mmc_text1.py` | Evaluates MMC‑Text predictions |

### **Run MMC-MQA Benchmark**
```bash
python benchmark/run_mmc_mqa.py
python benchmark/eval_mmc_mqa.py
```

### **Run MMC-Text Benchmark**
```bash
python benchmark/run_mmc_text_full2.py
python benchmark/eval_mmc_text1.py
```

---

## 🧪 Finetune Folder  
Directory: `finetune/`

Contains all scripts for Q‑LoRA fine‑tuning on MMC‑Instruction.

| File | Description |
|------|-------------|
| `train.py` | Main fine‑tuning script |
| `train_mem.py` | Memory‑efficient version |
| `to_run.py` | Launcher script for training |
| `final.py` | Final cleaned training pipeline |

### **Run Fine‑Tuning**
```bash
python finetune/to_run.py
```
or
```bash
python finetune/final.py
```

---

## 🚀 Quick Start Summary

```bash
# 1. Start Controller–Worker–UI
bash UI/controller.sh
bash UI/worker.sh
bash UI/web.sh

# 2. Run MMC-Text Benchmark
python benchmark/run_mmc_text_full2.py
python benchmark/eval_mmc_text1.py

# 3. Run MMC-MQA Benchmark
python benchmark/run_mmc_mqa.py
python benchmark/eval_mmc_mqa.py

# 4. Fine-Tune the Model
python finetune/to_run.py
```

---


### 📚 Dataset
We use the **MMC Dataset** released with the NAACL 2024 paper:
👉 https://huggingface.co/datasets/xywang1/MMC

It includes:
- MMC-Alignment
- MMC-Instruction
- MMC-Benchmark (Text + MQA)

### 🤖 Base Model
The project is built on top of the **mPLUG-Owl2** multimodal LLM:
👉 https://github.com/X-PLUG/mPLUG-Owl/tree/main/mPLUG-Owl2

It provides:
- Vision encoder (ViT-L)
- Vicuna-7B language model
- Multimodal fusion architecture


## 📄 Citation

If you use this project, please cite the original MMC dataset and mPLUG-Owl2 model:

**MMC Dataset**
Liu et al., "MMC: Advancing Multimodal Chart Understanding with Large-scale Instruction Tuning", NAACL 2024.

**mPLUG-Owl2**
Ye et al., "mPLUG-Owl2: Multi-modal Large Language Model", 2023.




