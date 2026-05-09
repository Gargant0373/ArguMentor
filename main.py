def main():
    from src.baseline import run_baseline
    from src.baseline2 import run_baseline2

    results1 = run_baseline()
    results2 = run_baseline2()

    print("\n" + "=" * 80)
    print("Argument Quality Baseline Comparison (Test Set)")
    print("=" * 80 + "\n")

    metrics = ["accuracy", "f1_macro"]
    header = f"{'Metric':<15} {'TF-IDF + LogReg':<20} {'TF-IDF + LinearSVM':<20}"
    print(header)
    print("-" * 80)

    test1 = results1["test"]
    test2 = results2["test"]

    for metric in metrics:
        val1 = f"{test1[metric]:.4f}"
        val2 = f"{test2[metric]:.4f}"
        print(f"{metric:<15} {val1:<20} {val2:<20}")

    print("\n" + "=" * 80)
    print("Confusion Matrices (Test Set)")
    print("=" * 80)
    print("\nTF-IDF + Logistic Regression:")
    print(f"  {test1['labels']}")
    for i, row in enumerate(test1["confusion_matrix"]):
        print(f"  {test1['labels'][i]}: {row}")

    print("\nTF-IDF + Linear SVM:")
    print(f"  {test2['labels']}")
    for i, row in enumerate(test2["confusion_matrix"]):
        print(f"  {test2['labels'][i]}: {row}")
    print("=" * 80 + "\n")

if __name__ == "__main__":
    main()