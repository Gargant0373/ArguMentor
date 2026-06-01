# ArguMentor

An argument-quality assistant that classifies arguments as **low**, **medium**, or **high** quality and provides targeted, actionable feedback for improvement.

---

## Requirements

- Python 3.10+
- [Ollama](https://ollama.com) with `llama3.2:3b` pulled (for the feedback component)

---

## Setup

```bash
# 1. Clone the repository and enter it
git clone <repo-url>
cd Argumentor

# 2. Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt
```

---

## Running the Web UI

Make sure Ollama is running before launching the app:

```bash
ollama pull llama3.2:3b   # only needed once
ollama serve              # start the local LLM server
```

Then launch the Gradio interface:

```bash
python app.py
```

Open [http://localhost:7860](http://localhost:7860) in your browser.

### How to use

1. Enter the **topic** of the debate.
2. Select the **stance** (pro or con).
3. Type the **argument** you want to evaluate.
4. Choose a **classifier** — RoBERTa (fine-tuned) for best accuracy, or TF-IDF + Logistic Regression for a faster result.
5. Click **Analyse**.

The app returns a quality label and a structured feedback card with a focus area (Evidence, Clarity, Structure, Relevance, or Logic), a concrete suggestion, and an explanation of why it helps.

> **Note:** The RoBERTa model is loaded from the `./argument_model-roberta` cache directory. If the cache does not exist, the model will be trained from scratch on first launch — this takes a significant amount of time.

---

## Running the Baseline / Fine-tune Pipeline

To reproduce the full classification evaluation:

```bash
python main.py
```

This trains (or loads from cache) the TF-IDF + Logistic Regression baseline, the TF-IDF + LinearSVM baseline, and the fine-tuned RoBERTa model, then prints accuracy, macro-F1, and confusion matrices for each on the test set.

---

## Project Structure

```
app.py                  # Gradio web UI
main.py                 # CLI pipeline comparison
requirements.txt
src/
    dataset.py          # Data loading, cleaning, label construction
    baseline.py         # TF-IDF + Logistic Regression
    baseline2.py        # TF-IDF + Linear SVM
    finetune.py         # RoBERTa fine-tuning pipeline
    predictor.py        # Single-item inference wrapper (used by the UI)
    feedback.py         # Ollama-based feedback generation
```

---

## Dataset

The project uses the [IBM Argument Quality Ranking 30k](https://huggingface.co/datasets/ibm-research/argument_quality_ranking_30k) dataset. It is downloaded automatically via the Hugging Face Hub on first run.
