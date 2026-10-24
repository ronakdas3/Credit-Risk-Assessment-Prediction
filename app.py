"""
FastAPI Microservice for Real-Time Credit Risk & Default Prediction
"""
import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Optional

app = FastAPI(
    title="Credit Risk Assessment & Prediction API",
    description="Real-time loan default prediction using optimized Scikit-learn Pipeline",
    version="1.0.0"
)

MODEL_PATH = "models/best_credit_risk_model.joblib"
pipeline = None

@app.on_event("startup")
def load_model():
    global pipeline
    try:
        pipeline = joblib.load(MODEL_PATH)
        print("Model pipeline loaded successfully.")
    except Exception as e:
        print(f"Warning: Model not found at {MODEL_PATH}. Run src/benchmark.py first. ({e})")

class ApplicantData(BaseModel):
    person_age: int = Field(..., example=32)
    person_income: float = Field(..., example=65000.0)
    person_home_ownership: str = Field(..., example="RENT")
    person_emp_length: Optional[float] = Field(None, example=4.0)
    loan_intent: str = Field(..., example="DEBTCONSOLIDATION")
    loan_amnt: float = Field(..., example=12000.0)
    loan_term_months: int = Field(..., example=36)
    credit_score: int = Field(..., example=670)
    debt_to_income_ratio: float = Field(..., example=24.5)
    revolving_utilization: Optional[float] = Field(None, example=45.2)
    delinq_2yrs: int = Field(..., example=0)
    inq_last_6mths: int = Field(..., example=1)

@app.get("/")
def health_check():
    return {"status": "healthy", "service": "Credit Risk Prediction Service"}

@app.post("/predict")
def predict_default(applicant: ApplicantData):
    global pipeline
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Model is not yet loaded.")
    
    # Format input as a DataFrame
    input_df = pd.DataFrame([applicant.model_dump()])
    
    # Inference
    proba = pipeline.predict_proba(input_df)[0][1]
    prediction = int(proba >= 0.5)
    risk_tier = "High Risk" if proba > 0.65 else ("Moderate Risk" if proba > 0.35 else "Low Risk")
    
    return {
        "default_probability": round(float(proba), 4),
        "predicted_default": prediction,
        "risk_tier": risk_tier,
        "recommendation": "Reject" if proba >= 0.5 else "Approve"
    }

if __name__ == '__main__':
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
