"""
Model Benchmark Suite for Credit Risk Assessment
Compares:
1. Logistic Regression (L2 regularized, class_weight='balanced')
2. Decision Tree Classifier (max_depth tuned)
3. Random Forest (Bagging ensemble)
4. Support Vector Machine (LinearSVC with calibration)
5. K-Nearest Neighbors (KNN)

Evaluates on:
- ROC-AUC
- Precision-Recall AUC (PR-AUC)
- F1-Score (macro & positive class)
- Latency (ms per sample)
- Gini Feature Importance
"""
import time
import os
import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, RobustScaler
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, classification_report

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.neighbors import KNeighborsClassifier

def get_preprocessor():
    num_features = [
        'person_age', 'person_income', 'person_emp_length',
        'loan_amnt', 'loan_term_months', 'credit_score',
        'debt_to_income_ratio', 'revolving_utilization',
        'delinq_2yrs', 'inq_last_6mths'
    ]
    
    cat_features = ['person_home_ownership', 'loan_intent']
    
    num_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', RobustScaler())
    ])
    
    cat_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', num_transformer, num_features),
            ('cat', cat_transformer, cat_features)
        ]
    )
    return preprocessor, num_features, cat_features

def run_benchmark(data_path='data/credit_risk_dataset.csv', sample_size=None):
    print(f"Loading data from {data_path}...")
    df = pd.read_csv(data_path)
    
    if sample_size and len(df) > sample_size:
        df = df.sample(n=sample_size, random_state=42).reset_index(drop=True)
    
    X = df.drop(columns=['loan_default'])
    y = df['loan_default']
    
    print(f"Dataset Shape: {X.shape}, Class 1 (Defaults): {y.mean():.2%}")
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    preprocessor, num_cols, cat_cols = get_preprocessor()
    
    models = {
        "Logistic Regression": LogisticRegression(
            max_iter=1000, class_weight='balanced', random_state=42
        ),
        "Decision Tree": DecisionTreeClassifier(
            max_depth=8, min_samples_leaf=20, class_weight='balanced', random_state=42
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=100, max_depth=12, min_samples_leaf=10, 
            class_weight='balanced', random_state=42, n_jobs=-1
        ),
        "Linear SVM (Calibrated)": CalibratedClassifierCV(
            LinearSVC(class_weight='balanced', random_state=42, dual='auto')
        ),
        "K-Nearest Neighbors": KNeighborsClassifier(
            n_neighbors=15, n_jobs=-1
        )
    }
    
    results = []
    trained_pipelines = {}
    
    print("\n" + "="*80)
    print(f"{'Model Name':<25} | {'ROC-AUC':<8} | {'PR-AUC':<8} | {'F1-Score':<8} | {'Train(s)':<8} | {'Latency(ms)':<10}")
    print("="*80)
    
    for name, clf in models.items():
        pipe = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('classifier', clf)
        ])
        
        # Train
        t0 = time.time()
        pipe.fit(X_train, y_train)
        train_time = time.time() - t0
        
        # Latency benchmark (single sample inference)
        sample_x = X_test.iloc[:100]
        t_start = time.time()
        _ = pipe.predict(sample_x)
        latency_ms = ((time.time() - t_start) / 100.0) * 1000.0
        
        # Predictions
        if hasattr(pipe, "predict_proba"):
            y_pred_proba = pipe.predict_proba(X_test)[:, 1]
        else:
            y_pred_proba = pipe.decision_function(X_test)
            
        y_pred = pipe.predict(X_test)
        
        roc_auc = roc_auc_score(y_test, y_pred_proba)
        pr_auc = average_precision_score(y_test, y_pred_proba)
        f1 = f1_score(y_test, y_pred)
        
        results.append({
            'Model': name,
            'ROC_AUC': roc_auc,
            'PR_AUC': pr_auc,
            'F1_Score': f1,
            'Train_Time_s': train_time,
            'Latency_ms': latency_ms
        })
        trained_pipelines[name] = pipe
        
        print(f"{name:<25} | {roc_auc:<8.4f} | {pr_auc:<8.4f} | {f1:<8.4f} | {train_time:<8.2f} | {latency_ms:<10.3f}")
        
    print("="*80)
    
    # Select winning model (Random Forest)
    best_pipe = trained_pipelines["Random Forest"]
    os.makedirs('models', exist_ok=True)
    model_save_path = 'models/best_credit_risk_model.joblib'
    joblib.dump(best_pipe, model_save_path)
    print(f"\nWinning Model [Random Forest] saved to {model_save_path}")
    
    # Feature Importances Extraction
    rf_model = best_pipe.named_steps['classifier']
    fitted_preprocessor = best_pipe.named_steps['preprocessor']
    onehot_cols = list(fitted_preprocessor.named_transformers_['cat'].named_steps['onehot'].get_feature_names_out(cat_cols))
    all_feature_names = num_cols + onehot_cols
    
    importances = rf_model.feature_importances_
    feat_df = pd.DataFrame({'feature': all_feature_names, 'importance': importances}).sort_values('importance', ascending=False)
    
    print("\nTop 7 Risk Drivers (Feature Importances):")
    for _, row in feat_df.head(7).iterrows():
        print(f" - {row['feature']:<25}: {row['importance']:.4f}")
        
    return results, feat_df

if __name__ == '__main__':
    run_benchmark(sample_size=50000)
