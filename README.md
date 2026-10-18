# NewsGuard ML Studio 2.2

A portfolio-focused, educational **English-language text classification laboratory** developed from an undergraduate machine-learning project. It demonstrates supervised NLP, model evaluation, explainability and CSV batch processing.

> **Important:** This model recognizes patterns in historical labeled news texts. It **does not verify facts** or consult sources. The model scores are not calibrated probabilities of factual correctness. Do not use for consequential moderation decisions.

## Features

- **Analyze article:** individual text prediction with token contribution charts and descriptive vocabulary coverage.
- **Batch CSV analysis:** UTF-8 CSV uploads up to 10 MB / 5,000 data records, text-column selector, five-row input preview, result counts (TRUE/FAKE/skipped), downloadable results retaining the original text and all source columns.
- **Model dashboard:** accuracy, precision, recall, F1, ROC-AUC, confusion matrix and ROC curve.
- **Challenge set:** manually constructed examples demonstrating likely failure cases, not a representative independent benchmark.
- **Tests:** checks for deduplication, model explanations, CSV parsing, inference, export and validation.

## Install and launch (macOS / Linux)

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m src.pipeline
python -m pytest -q
python -m streamlit run app.py
```

Open the URL displayed by Streamlit (usually http://localhost:8501).

Training expects `data/true.csv` and `data/fake.csv`, containing `title` and `text` columns. A **Lite** archive excludes these files; bring your authorized copies of them into `data/` before training.

## Expected project layout

```text
app.py
src/
  __init__.py
  batch.py
  insights.py
  pipeline.py
tests/
  test_batch.py
  test_insights.py
  test_pipeline.py
data/
  true.csv        # not included in the Lite archive
  fake.csv        # not included in the Lite archive
artifacts/       # generated locally, ignored by Git
requirements.txt
```

The training script writes `artifacts/model.joblib`, `artifacts/metrics.json`, `artifacts/confusion_matrix.png` and `artifacts/roc_curve.png`.

## Validation and limitations

The original supplied news data produced approximately **98.22% holdout accuracy** with a stratified random 80/20 split after deduplication. This is **within-dataset** evaluation, not evidence of performance on independent sources or current events. Source-specific language patterns, dataset composition, temporal drift and out-of-distribution inputs may affect results. The challenge set demonstrates some failures but is not a statistically valid benchmark.

## Screenshots

Add actual screenshots captured locally from the application before publishing; screenshots are not fabricated or bundled.

## Privacy, data and publishing

- CSV files are processed by the running Streamlit server. When locally hosted on your computer this means local processing, but deployments to a cloud server handle files on that remote server.
- Do not upload confidential or unauthorized material.
- **Check dataset licensing before redistribution.** The full local archive contains the user-provided datasets; `.gitignore` excludes `data/*.csv` from Git commits.
- Avoid committing model artifacts to a public Git repository until provenance and redistribution rights are confirmed.

## Tech stack

Python, pandas, NumPy, scikit-learn, Streamlit, matplotlib, joblib, pytest.
