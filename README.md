# Credit Risk Assessment & Prediction Suite

An end-to-end Classical Machine Learning benchmark and real-time inference microservice evaluating:
- Logistic Regression
- K-Nearest Neighbors (KNN)
- Support Vector Machine (Linear & RBF)
- Decision Tree
- Random Forest

## Structure
- `src/data_generator.py`: Generates realistic synthetic credit risk / loan default records (150k+ samples) with non-linear correlations, missing values, and imbalanced classes.
- `src/benchmark.py`: Runs 5-fold Stratified Cross-Validation across candidate models, metrics (ROC-AUC, PR-AUC, F1, Latency), extracts feature importances, and exports champion model.
- `app.py`: FastAPI serving endpoint for real-time risk scoring.
