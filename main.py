import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import numpy as np
from typing import List, Optional, Dict
from datetime import datetime

# Initialize FastAPI app
app = FastAPI(title="Enhanced Emotional Wellbeing API", version="5.0")

# ---------------- Models ----------------
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
    confidence_score: float
    wellbeing_score: int
    wellbeing_breakdown: Dict[str, float]
    recommendations: List[SmartRecommendation]
    risk_factors: List[str]
    positive_factors: List[str]
    next_check_in: str

# ---------------- Load Model ----------------
try:
    model = joblib.load('models/emotion_model_v5.joblib')
    scaler = joblib.load('models/scaler_v5.joblib')
    label_encoder = joblib.load('models/label_encoder_v5.joblib')
    model_features = joblib.load('models/features_v5.joblib')
except FileNotFoundError:
    raise RuntimeError("Model files not found in 'models/' folder.")

# ---------------- Engine & Scorer ----------------
class SmartRecommendationEngine:
    def __init__(self):
        self.recommendation_templates = {
            'sleep': {
                'insufficient': ["Try a consistent bedtime routine."],
                'excessive': ["Avoid oversleeping to maintain rhythm."]
            },
            'activity': {
                'low': ["Try a 10-minute walk today."],
                'optimal': ["Great activity level!"]
            },
            'social_media': {
                'excessive': ["Use app timers to reduce screen time."],
                'platform_specific': ["Instagram usage is high."]
            },
            'music': {
                'restless': ["Try a focus playlist to reduce skipping."],
                'mood_mismatch': ["Use mood-based playlists."]
            },
            'physiological': {
                'high_hr': ["Try deep breathing if heart rate is high."],
                'optimal_hr': ["Heart rate is in a healthy range."]
            }
        }

    def analyze_context(self, data: PredictionInput) -> Dict:
        context = {}
        hour = data.currentHour or datetime.now().hour
        if hour < 6 or hour >= 22:
            context['time_period'] = 'night'
        elif hour < 12:
            context['time_period'] = 'morning'
        elif hour < 18:
            context['time_period'] = 'afternoon'
        else:
            context['time_period'] = 'evening'

        if data.stepsCount < 3000:
            context['activity_level'] = 'sedentary'
        elif data.stepsCount < 8000:
            context['activity_level'] = 'moderate'
        else:
            context['activity_level'] = 'active'

        if data.sleepHours < 6:
            context['sleep_quality'] = 'insufficient'
        elif data.sleepHours > 9:
            context['sleep_quality'] = 'excessive'
        else:
            context['sleep_quality'] = 'adequate'

        return context

    def calculate_impact_score(self, category: str, data: PredictionInput) -> float:
        return round(np.random.uniform(0.4, 0.9), 2)

    def generate_smart_recommendations(self, data: PredictionInput, emotion: str) -> List[SmartRecommendation]:
        context = self.analyze_context(data)
        recs = []

        if data.sleepHours < 6:
            recs.append(SmartRecommendation(
                category="Sleep",
                text=self.recommendation_templates['sleep']['insufficient'][0],
                priority=1,
                impact_score=self.calculate_impact_score('sleep', data),
                time_to_implement="Tonight"
            ))

        if data.stepsCount < 4000:
            recs.append(SmartRecommendation(
                category="Activity",
                text=self.recommendation_templates['activity']['low'][0],
                priority=2,
                impact_score=self.calculate_impact_score('activity', data),
                time_to_implement="Today"
            ))

        if data.socialTime > 240:
            recs.append(SmartRecommendation(
                category="Digital Wellness",
                text=self.recommendation_templates['social_media']['excessive'][0],
                priority=3,
                impact_score=self.calculate_impact_score('social_media', data),
                time_to_implement="Today"
            ))

        if data.heartRate > 90:
            recs.append(SmartRecommendation(
                category="Physiological",
                text=self.recommendation_templates['physiological']['high_hr'][0],
                priority=2,
                impact_score=self.calculate_impact_score('physiological', data),
                time_to_implement="Now"
            ))

        return recs[:4]

class EnhancedWellbeingScorer:
    def calculate_comprehensive_score(self, data: PredictionInput) -> tuple:
        sleep_score = max(0, 100 - abs(data.sleepHours - 7.5) * 10)
        steps_score = min(100, data.stepsCount / 100)
        heart_score = 100 if 60 <= data.heartRate <= 75 else 50
        music_score = data.avg_valence * 100
        social_score = max(0, 100 - data.socialTime / 3)

        final = int(0.3 * sleep_score + 0.2 * steps_score + 0.2 * heart_score +
                    0.15 * music_score + 0.15 * social_score)

        breakdown = {
            'sleep_quality': round(sleep_score, 1),
            'physical_activity': round(steps_score, 1),
            'physiological_health': round(heart_score, 1),
            'music_mood': round(music_score, 1),
            'digital_wellness': round(social_score, 1)
        }

        return final, breakdown

# ---------------- Helpers ----------------
def identify_risk_and_positive_factors(data: PredictionInput):
    risks, positives = [], []

    if data.sleepHours < 6:
        risks.append("Low sleep")
    if data.stepsCount < 3000:
        risks.append("Low activity")
    if data.socialTime > 300:
        risks.append("High social media usage")

    if 7 <= data.sleepHours <= 8.5:
        positives.append("Good sleep")
    if data.stepsCount > 7000:
        positives.append("Active lifestyle")

    return risks, positives

def determine_next_checkin(predicted_emotion: str, risk_factors: List[str]):
    if predicted_emotion in ['Stressed', 'Anxious'] or len(risk_factors) > 2:
        return "Check back in 4–6 hours"
    if len(risk_factors) > 0:
        return "Check tomorrow"
    return "Check in 2–3 days"

# ---------------- API Routes ----------------
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
