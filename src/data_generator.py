"""
Data Generator for Credit Risk & Loan Default Prediction
Generates a realistic 150,000+ row dataset with:
- Non-linear relationships between financial variables
- Class imbalance (~15% default rate)
- Realistic missingness and categorical distributions
"""
import numpy as np
import pandas as pd
import os

def generate_credit_data(n_samples=150000, random_seed=42):
    np.random.seed(random_seed)
    
    # 1. Demographics & Credit Profile
    age = np.random.normal(loc=38, scale=11, size=n_samples).clip(20, 75).astype(int)
    annual_inc = np.random.lognormal(mean=10.8, sigma=0.6, size=n_samples).round(-2)
    emp_length = np.random.choice([0, 1, 2, 3, 5, 7, 10], size=n_samples, p=[0.1, 0.08, 0.09, 0.12, 0.16, 0.15, 0.3])
    home_ownership = np.random.choice(['RENT', 'MORTGAGE', 'OWN', 'OTHER'], size=n_samples, p=[0.42, 0.45, 0.12, 0.01])
    
    # 2. Loan Characteristics
    loan_amnt = np.random.choice(np.arange(1000, 40000, 500), size=n_samples)
    loan_intent = np.random.choice(['DEBTCONSOLIDATION', 'EDUCATION', 'HOMEIMPROVEMENT', 'MEDICAL', 'PERSONAL', 'VENTURE'], size=n_samples)
    term_months = np.random.choice([36, 60], size=n_samples, p=[0.72, 0.28])
    
    # 3. Credit Scores and Indicators
    credit_score = np.random.normal(loc=680, scale=60, size=n_samples).clip(450, 850).astype(int)
    revol_util = np.random.beta(a=2, b=3, size=n_samples) * 100
    dti = (loan_amnt / (annual_inc + 1e-5)) * 100 + np.random.normal(loc=12, scale=5, size=n_samples).clip(2, 60)
    delinq_2yrs = np.random.poisson(lam=0.35, size=n_samples).clip(0, 8)
    inq_last_6mths = np.random.poisson(lam=0.6, size=n_samples).clip(0, 6)
    
    # Non-linear probability of default logit calculation
    # High DTI, low credit score, high loan amnt vs income, delinquencies increase default risk
    # Intercept adjusted so base default rate is realistic (~15-18% imbalanced)
    logit = (
        -6.3
        + 0.045 * dti
        - 0.009 * (credit_score - 600)
        + 0.02 * revol_util
        + 0.35 * delinq_2yrs
        + 0.18 * inq_last_6mths
        + (loan_amnt / (annual_inc + 1e-5)) * 1.8
        - 0.03 * emp_length
    )
    
    # Interaction terms: Low credit score combined with high DTI exponentially compounds risk
    logit += ((credit_score < 620) & (dti > 25)) * 0.8
    
    prob_default = 1 / (1 + np.exp(-logit))
    loan_status = (np.random.uniform(0, 1, size=n_samples) < prob_default).astype(int)
    
    df = pd.DataFrame({
        'person_age': age,
        'person_income': annual_inc,
        'person_home_ownership': home_ownership,
        'person_emp_length': emp_length.astype(float),
        'loan_intent': loan_intent,
        'loan_amnt': loan_amnt,
        'loan_term_months': term_months,
        'credit_score': credit_score,
        'debt_to_income_ratio': dti.round(2),
        'revolving_utilization': revol_util.round(2),
        'delinq_2yrs': delinq_2yrs,
        'inq_last_6mths': inq_last_6mths,
        'loan_default': loan_status
    })
    
    # Introduce ~3% realistic missing values in emp_length & credit_score
    nan_mask_emp = np.random.rand(n_samples) < 0.03
    df.loc[nan_mask_emp, 'person_emp_length'] = np.nan
    
    nan_mask_revol = np.random.rand(n_samples) < 0.02
    df.loc[nan_mask_revol, 'revolving_utilization'] = np.nan
    
    return df

if __name__ == '__main__':
    os.makedirs('data', exist_ok=True)
    out_path = 'data/credit_risk_dataset.csv'
    print(f"Generating 150,000 credit records...")
    df = generate_credit_data(150000)
    df.to_csv(out_path, index=False)
    print(f"Dataset saved to {out_path} ({len(df)} rows). Default rate: {df['loan_default'].mean():.2%}")
