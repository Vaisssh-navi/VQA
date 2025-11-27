import json

PRED_FILE = "pred_mmc_mqa.jsonl"   # <-- change filename if needed

def normalize(s):
    """Normalize text for case-insensitive comparison."""
    if s is None:
        return ""
    return s.strip().lower()

def main():
    total = 0
    correct = 0

    with open(PRED_FILE, "r", encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)

            gt = normalize(item.get("gt", ""))
            pred = normalize(item.get("pred", ""))

            # Basic correctness check
            is_correct = (gt == pred)

            total += 1
            if is_correct:
                correct += 1

    acc = correct / total * 100 if total > 0 else 0.0

    print("=======================================")
    print(f"MQA Evaluation Results")
    print("=======================================")
    print(f"Total Samples  : {total}")
    print(f"Correct        : {correct}")
    print(f"Accuracy       : {acc:.2f}%")
    print("=======================================")

if __name__ == "__main__":
    main()

