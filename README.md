# Atrial Fibrillation Detection from 12-Lead ECG

Exploratory analysis and a Random Forest classifier for spotting atrial fibrillation (AF) in 12-lead ECG recordings, built on a PTB-XL–based dataset.

## Why this project

Atrial fibrillation is an irregular, often rapid heartbeat. It raises the risk of stroke and heart failure, and a good share of patients (roughly 15–30%) have no symptoms at all. Catching it early makes a big difference: early detection and proper management can cut stroke risk by about two-thirds.

This notebook looks at what the data tells us about AF patients, then trains a model that uses the raw ECG signal together with basic patient information to tell three rhythm classes apart.

## Dataset

The analysis uses the **PTB-XL Atrial Fibrillation Detection** dataset (a Kaggle dataset derived from the PTB-XL ECG database).

| File | What it holds |
|---|---|
| `coorteeqsrafva.csv` | Patient and recording metadata (age, sex, height, weight, heart axis, rhythm label, and so on). 6,428 rows. |
| `ecgeq-500hzsrfava.npy` | The ECG signals as a 3D array of shape `(6428, 5000, 12)`: 6,428 recordings, 5,000 time steps (500 Hz), 12 leads. |
| `af_dataset.csv` | A flattened, model-ready table (about 4.3 million rows, 26 columns) built from the two files above. It is created in a separate feature-engineering notebook, linked in the notebook. |

**Rhythm classes (`ritmi` column)**

| Label | Meaning | Rows |
|---|---|---|
| `SR` (0) | Normal sinus rhythm | 2,000 |
| `AF` (1) | Atrial fibrillation | 1,587 |
| `VA` (2) | Other arrhythmia | 2,841 |

**The 12 leads:** I, II, III, aVR, aVL, aVF (limb leads) and V1–V6 (precordial leads).

## What's in the notebook

1. **Preprocessing**
   - Dropped columns that are not useful for analysis (IDs, report text, noise flags, file names, etc.).
   - Encoded the rhythm label (`SR`→0, `AF`→1, `VA`→2) and the `validated_by_human` flag (`False`→0, `True`→1).
   - Grouped age, height and weight into bands, and pulled the recording year out of the recording date.

2. **Exploratory data analysis** — five questions, each answered with a count plot:
   - Which sex shows more AF cases?
   - Which age group shows more AF cases?
   - What weight range is most common among AF patients?
   - What height range is most common among AF patients?
   - What is the most common heart axis among AF patients?

   The notebook also plots five leads of a randomly picked normal, AF and other-arrhythmia ECG, so you can compare the shapes by eye.

3. **Modeling**
   - Features: the 12 ECG leads plus patient/recording fields such as `age`, `sex`, `height`, `weight`, `heart_axis`, `site`, `device`, `validated_by_human` and `strat_fold`.
   - Target: `ritmi`.
   - Split: 75% train / 25% test (`random_state=1234`).
   - Model: `RandomForestClassifier`, tuned with `GridSearchCV` (7-fold cross-validation). The grid in the notebook uses `n_estimators=45`, `criterion='entropy'`, `max_depth=45`.
   - Evaluation: `classification_report` on the held-out test set.

## Key observations from the EDA

These are read directly off the plots in the notebook:

- AF appears more often among **male** patients than female patients in this data.
- The **70–89** age range shows the most AF cases.
- Most AF patients weigh **under 60 kg or between 60 and 79 kg**, and are **1.50–1.79 m** tall.
- Most AF patients have a **normal heart electrical axis**.

## Results

The modeling cells need `af_dataset.csv`, which was not available when this notebook was last saved, so no scores are stored in the file. Run the notebook and fill in your numbers here:

| Class | Precision | Recall | F1-score |
|---|---|---|---|
| SR (normal) | | | |
| AF | | | |
| VA (other) | | | |

Best parameters found by grid search: `{n_estimators: 45, criterion: 'entropy', max_depth: 45}`

## Getting started

**1. Clone the repo**

```bash
git clone https://github.com/<your-username>/<your-repo-name>.git
cd <your-repo-name>
```

**2. Install the requirements**

```bash
pip install numpy pandas matplotlib seaborn scikit-learn jupyter
```

**3. Get the data**

Download the PTB-XL Atrial Fibrillation Detection dataset from Kaggle and generate `af_dataset.csv` using the feature-engineering notebook.

**4. Fix the file paths**

The notebook was written on Kaggle, so the paths look like this:

```python
pd.read_csv('../input/ptbxl-atrial-fibrillation-detection/coorteeqsrafva.csv', sep=';', header=0, index_col=0)
np.load('../input/ptbxl-atrial-fibrillation-detection/ecgeq-500hzsrfava.npy')
pd.read_csv('../input/af-dataset/af_dataset.csv')
```

Change them to wherever you saved the files on your machine.

**5. Run it**

```bash
jupyter notebook
```

Open the notebook and run the cells from top to bottom.

## Things to keep in mind

- The EDA plots show **counts**, not risk. A group with more bars simply has more rows in this dataset. The classes are also different sizes, and some recordings appear in more than one row (a recording with several diagnoses is listed once per diagnosis). Treat the EDA findings as descriptions of this dataset, not medical conclusions.
- The train/test split is done on **individual rows** of the flattened table. Because each recording is expanded into many rows, rows from the same recording can end up in both the training and test sets, which can make scores look better than they really are. Splitting by recording or patient (for example with `patient_id` or `strat_fold`) is a fairer test.
- Some features (`site`, `device`, `validated_by_human`, `strat_fold`) describe how the data was collected rather than the patient's heart. It is worth checking how much the model leans on them.
- This is a learning and research project. It is **not** a medical device and should not be used to diagnose anyone.

## Tech stack

Python · NumPy · pandas · Matplotlib · seaborn · scikit-learn

## Acknowledgements

- **PTB-XL** ECG database (PhysioNet) and the Kaggle *PTB-XL Atrial Fibrillation Detection* dataset built from it.
- The feature-engineering notebook that creates `af_dataset.csv` is linked inside the notebook (`tvo10/atrial-fibrillation-detection` on GitHub). If you built on that work, please keep the credit here.
