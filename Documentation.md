# AI-Based Diabetes Prediction System

## Dataset
- **Source:** Kaggle — Diabetes Health Indicators Dataset (BRFSS 2015)
- **File:** diabetes_binary_health.csv
- **Records:** ~250,000 patient entries
- **Features:** 21 health indicators (BMI, Age, Blood Pressure, Cholesterol, etc.)
- **Target:** Diabetes_binary (0 = Non-Diabetic, 1 = Diabetic)

## Preprocessing Techniques
- Handled missing values using column median imputation
- Normalized features using StandardScaler
- Stratified train/test split (80/20) to preserve class balance

## Model Training
- Algorithm: Random Forest Classifier
- Parameters: 100 trees, max depth 10, min samples split 5
- Training set: 80% of data (~200,000 samples)
- Test set: 20% of data (~50,000 samples)

## Evaluation Results
| Metric    | Score  |
|-----------|--------|
| Accuracy  | ~85%   |
| Precision | ~62%   |
| Recall    | ~10%   |
| F1-Score  | ~17%   |
| ROC-AUC   | ~0.8  |



