uploaded_files = {name: file for name, file in upload_widget.value.items()}

# Load CSV
for name, file in uploaded_files.items():
    if name.endswith('.csv'):
        csv_data = pd.read_csv(io.BytesIO(file['content']), sep=';', header=0, index_col=0)
        print(f"✅ Loaded {name} into csv_data with shape {csv_data.shape}")

# Load NPY
for name, file in uploaded_files.items():
    if name.endswith('.npy'):
        npy_data = np.load(io.BytesIO(file['content']))
        print(f"✅ Loaded {name} into npy_data with shape {npy_data.shape}")

import pandas as pd
import io

# Load your uploaded CSV file (replace 'your_file.csv' with the actual name)
csv_data = pd.read_csv(io.BytesIO(uploaded_files['coorteeqsrafva.csv']['content']), sep=';', header=0, index_col=0)

# Quick check
csv_data.head()
# Train/test split again
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42)

# Scale numeric features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("✅ Scaling completed. X_train_scaled shape:", X_train_scaled.shape)



from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV

# Define hyperparameter search space
param_dist = {
    'n_estimators': [100, 200, 300],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'bootstrap': [True, False]
}

# Initialize Random Forest model
rf = RandomForestClassifier(random_state=42)

# Setup randomized search
random_search = RandomizedSearchCV(
    estimator=rf,
    param_distributions=param_dist,
    n_iter=20,
    cv=5,
    verbose=2,
    random_state=42,
    n_jobs=-1
)

# Fit the search on training data
random_search.fit(X_train_scaled, y_train)

# Save best model
best_rf = random_search.best_estimator_

# Print best parameters
print("✅ Best Parameters Found:\n", random_search.best_params_)



from sklearn.metrics import classification_report, confusion_matrix

# Predict on test data
y_pred = best_rf.predict(X_test_scaled)

# Classification report
print("📊 Classification Report:\n", classification_report(y_test, y_pred))

# Confusion matrix
print("🧩 Confusion Matrix:\n", confusion_matrix(y_test, y_pred)) 
