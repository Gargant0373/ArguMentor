def main():
    from src.baseline import run_baseline
    from src.baseline2 import run_baseline2
    from src.fewshot_ollama import run_fewshot_ollama
    from src.finetune import run_finetune
    from src.zeroshot_ollama import run_zeroshot_ollama

    results1 = run_baseline()
    results2 = run_baseline2()
    results3 = run_finetune()
    results4 = run_zeroshot_ollama()
    results5 = run_fewshot_ollama()

    print("\n" + "=" * 120)
    print("Argument Quality Pipeline Comparison (Test Set)")
    print("=" * 120 + "\n")

    metrics = ["accuracy", "f1_macro", "mae", "off_by_one_acc", "qwk"]
    header = (
        f"{'Metric':<17} {'TF-IDF + LogReg':<20} {'TF-IDF + LinearSVM':<20}"
        f" {'RoBERTa FT':<20} {'Ollama Zero-Shot':<20} {'Ollama Few-Shot':<20}"
    )
    print(header)
    print("-" * 120)

    test1 = results1["test"]
    test2 = results2["test"]
    test3 = results3["test"]
    test4 = results4["test"]
    test5 = results5["test"]

    for metric in metrics:
        def _fmt(d: dict, m: str) -> str:
            return f"{d[m]:.4f}" if m in d else "N/A"
        print(
            f"{metric:<17} {_fmt(test1, metric):<20} {_fmt(test2, metric):<20}"
            f" {_fmt(test3, metric):<20} {_fmt(test4, metric):<20} {_fmt(test5, metric):<20}"
        )

    print("\n" + "=" * 120)
    print("Confusion Matrices / Classification Reports (Test Set)")
    print("=" * 120)
    print("\nTF-IDF + Logistic Regression:")
    print(f"  {test1['labels']}")
    for i, row in enumerate(test1["confusion_matrix"]):
        print(f"  {test1['labels'][i]}: {row}")

    print("\nTF-IDF + Linear SVM:")
    print(f"  {test2['labels']}")
    for i, row in enumerate(test2["confusion_matrix"]):
        print(f"  {test2['labels'][i]}: {row}")

    print("\nRoBERTa Fine-tuned:")
    print(test3["classification_report"])

    print("\nOllama Zero-Shot:")
    print(test4["classification_report"])

    k = results5["k_per_class"]
    print(f"\nOllama Few-Shot (k={k}/class):")
    print(test5["classification_report"])
    print("=" * 120 + "\n")

if __name__ == "__main__":
    main()