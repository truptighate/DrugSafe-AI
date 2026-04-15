import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from app.predict import predict_interaction
from app.explain import generate_explanation

app = FastAPI()

static_dir = os.path.join(os.path.dirname(__file__), "static")
app.mount("/static", StaticFiles(directory=static_dir), name="static")

class DrugPair(BaseModel):
    drug_a: str
    drug_b: str

@app.get("/")
def index():
    return FileResponse(os.path.join(static_dir, "index.html"))

@app.post("/analyze")
def analyze(pair: DrugPair):
    prediction = predict_interaction(pair.drug_a.strip(), pair.drug_b.strip())
    if 'error' in prediction:
        return {"error": prediction['error']}
    explanation = generate_explanation(prediction)
    return {
        "drug_a"      : prediction['drug_a'],
        "drug_b"      : prediction['drug_b'],
        "risk_level"  : prediction['risk_level'],
        "probabilities": prediction['probabilities'],
        "shap_features": prediction['top_shap_features'],
        "explanation" : explanation
    }