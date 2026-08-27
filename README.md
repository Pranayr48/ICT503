# ICT503 Cybersecurity Threat Detection

This project downloads and analyzes the Kaggle cybersecurity threat detection dataset, then trains a leakage-aware ExtraTrees classifier.

## Files

- `CybersecurityThreatDetection.ipynb`: EDA, feature engineering, training, and evaluation.
- `data/cybersecurity.csv`: downloaded dataset.
- `data/cybersecurity-threat-detection-dataset.zip`: original Kaggle archive.

## Results

Using a stratified 80/20 split with `random_state=42`:

- Training accuracy: **100.0%**
- Testing accuracy: **96.9%**
- ROC-AUC: **0.880**

`attack_type` is excluded from model features because it directly describes the target and would cause target leakage. The dataset is highly imbalanced (9,600 benign and 400 attacks), so accuracy alone is not sufficient: the test attack recall is 23.7% and balanced accuracy is 61.8%. These metrics should be improved before using the model for operational threat detection.

## Run

Open `CybersecurityThreatDetection.ipynb` in VS Code or Jupyter and run all cells. The notebook expects the CSV at `data/cybersecurity.csv`.
