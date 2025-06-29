import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import numpy as np
from typing import List, Optional, Dict
from datetime import datetime

# Initialize the FastAPI App
app = FastAPI(title="Enhanced Emotional Wellbeing API", version="5.0")

# ------------------------ Models ------------------------
class PredictionInput(BaseModel):
    sleepHours: float = Field(..., ge=0, le=24)
    stepsCount: int = Field(..., ge=0)
    caloriesBurnt: int = Field(..., ge=0)
    heartRate: int = Field(..., ge=30, le=200)
    songsSkipped: int = Field(..., ge=0)
    avg_valence: float = Field(..., ge=0, le=1)
    avg_energy: float = Field(..., ge=0, le=1)
    avg_danceability: float = Field(..., ge=0, le=1)
    socialTime: int = Field(...,)
    instagramTime: Optional[int] = Field(None, ge=0)
    xTime: Optional[int] = Field(None, ge=0)
    redditTime: Optional[int] = Field(None, ge=0)
    youtubeTime: Optional[int] = Field(None, ge=0)
    musicListeningTime: Optional[int] = Field(None, ge=0)
    currentHour: Optional[int] = Field(None, ge=0, le=23)

class SmartRecommendation(BaseModel):
    category: str
    text: str
    priority: int = Field(..., ge=1, le=5)
    actionable: bool = Field(True)
    impact_score: float = Field(..., ge=0, le=1)
    time_to_implement: str

class DetailedPrediction(BaseModel):
    predicted_emotion: str
    confidence_score: float = Field(..., ge=0, le=1)
    wellbeing_score: int = Field(..., ge=0, le=100)
    wellbeing_breakdown: Dict[str, float]
    recommendations: List[SmartRecommendation]
    risk_factors: List[str]
    positive_factors: List[str]
    next_check_in: str

# ------------------------ Load Artifacts ------------------------
try:
    model = joblib.load('models/emotion_model_v5.joblib')
    scaler = joblib.load('models/scaler_v5.joblib')
    label_encoder = joblib.load('models/label_encoder_v5.joblib')
    model_features = joblib.load('models/features_v5.joblib')
except FileNotFoundError:
    raise RuntimeError("Enhanced model artifacts not found.")

# ------------------------ Logic Modules ------------------------
# Paste all classes: SmartRecommendationEngine, EnhancedWellbeingScorer,
# and all functions like identify_risk_and_positive_factors, determine_next_checkin
# (To avoid repetition, reuse the code from your original logic. It remains unchanged.)

# For brevity, assume here it's already inserted exactly as you shared above

recommendation_engine = SmartRecommendationEngine()
wellbeing_scorer = EnhancedWellbeingScorer()

@app.post("/predict_enhanced", response_model=DetailedPrediction)
def predict_enhanced(input_data: PredictionInput):
    try:
        model_input_dict = {key: getattr(input_data, key) for key in model_features}
        input_df = pd.DataFrame([model_input_dict])
        input_scaled = scaler.transform(input_df)

        prediction_proba = model.predict_proba(input_scaled)[0]
        predicted_class = model.predict(input_scaled)[0]
        predicted_emotion = label_encoder.inverse_transform([predicted_class])[0]
        confidence_score = float(np.max(prediction_proba))

        wellbeing_score, breakdown = wellbeing_scorer.calculate_comprehensive_score(input_data)
        recommendations = recommendation_engine.generate_smart_recommendations(input_data, predicted_emotion)
        risks, positives = identify_risk_and_positive_factors(input_data)
        next_checkin = determine_next_checkin(predicted_emotion, risks)

        return DetailedPrediction(
            predicted_emotion=predicted_emotion,
            confidence_score=confidence_score,
            wellbeing_score=wellbeing_score,
            wellbeing_breakdown=breakdown,
            recommendations=recommendations,
            risk_factors=risks,
            positive_factors=positives,
            next_check_in=next_checkin
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@app.get("/health")
def health_check():
    return {"status": "healthy", "model_version": "v5_enhanced"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000)
