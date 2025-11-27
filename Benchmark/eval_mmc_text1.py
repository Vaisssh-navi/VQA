import json
from collections import defaultdict
from sklearn.metrics import classification_report, confusion_matrix

PRED_FILE = "pred_mmc_text1.jsonl"   # <-- your file
# PRED_FILE = "/home/ee/btech/ee1221719/mmc-work/mPLUG-Owl/mPLUG-Owl2/pred_mmc_text1.jsonl"

def clean(x):
    return x.strip().lower()

# -------------------------------
# LOAD PREDICTIONS
# -------------------------------
samples = []
with open(PRED_FILE, "r", encoding="utf-8") as f:
    for line in f:
        obj = json.loads(line)
        task = obj.get("task", "unknown")
        gt = clean(obj["gt"])
        pred = clean(obj["pred"])
        samples.append((task, gt, pred))

# -------------------------------
# OVERALL ACCURACY
# -------------------------------
total = len(samples)
correct = sum(1 for (_, gt, pred) in samples if gt == pred)
overall_acc = correct / total * 100

print("==============================================")
print(f"OVERALL ACCURACY: {overall_acc:.2f}% ({correct}/{total})")
print("==============================================\n")

# -------------------------------
# PER-TASK ACCURACY
# -------------------------------
task_stats = defaultdict(lambda: {"correct": 0, "total": 0})

for task, gt, pred in samples:
    task_stats[task]["total"] += 1
    if gt == pred:
        task_stats[task]["correct"] += 1

print("PER-TASK ACCURACY\n")
for task, stats in sorted(task_stats.items()):
    c = stats["correct"]
    t = stats["total"]
    acc = (c / t * 100) if t > 0 else 0
    print(f"{task:15s}: {acc:.2f}%   ({c}/{t})")
print()

# -------------------------------
# CLASSIFICATION REPORT
# -------------------------------
y_true = [gt for (_, gt, _) in samples]
y_pred = [pred for (_, _, pred) in samples]

print("\nClassification Report (true/false)\n")
print(classification_report(y_true, y_pred, digits=4))

# -------------------------------
# CONFUSION MATRIX
# -------------------------------
print("\nConfusion Matrix [GT rows, Pred columns]\n")
print(confusion_matrix(y_true, y_pred))

