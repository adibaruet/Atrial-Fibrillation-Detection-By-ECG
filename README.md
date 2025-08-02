# Atrial Fibrillation Detection with Machine Learning

This project uses Random Forest with hyperparameter tuning to classify ECG rhythms into:

- Normal Sinus Rhythm (SR)
- Atrial Fibrillation (AF)
- Other Arrhythmia (VA)

## Files

- `afib_rf_pipeline.py`: Main script for loading data, preprocessing, model training, and evaluation.
- `coorteeqsrafva.csv` and `ecgeq-500hzsrfava.npy`: ECG metadata and signal data (Kaggle-sourced).
- `README.md`: Project summary and usage info.

## Requirements

- pandas
- scikit-learn
- numpy

## Future Work

- Train a CNN model on raw ECG signals (`.npy` data)
- Add visualizations for ECG data
