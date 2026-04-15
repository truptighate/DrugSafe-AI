import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

client = OpenAI(
    base_url="https://integrate.api.nvidia.com/v1",
    api_key=os.getenv("NVIDIA_API_KEY")
)

def build_prompt(prediction: dict) -> str:
    drug_a     = prediction['drug_a']
    drug_b     = prediction['drug_b']
    risk       = prediction['risk_level']
    probs      = prediction['probabilities']
    shap_feats = prediction['top_shap_features']

    shap_lines = ""
    for f in shap_feats:
        shap_lines += f"  - {f['feature']}: {f['value']} (impact score: {f['shap_score']})\n"

    prompt = f"""You are a clinical pharmacist AI assistant. A machine learning model has 
predicted a drug-drug interaction. Your job is to explain this prediction clearly 
to a healthcare professional.

Drug A: {drug_a}
Drug B: {drug_b}
Predicted interaction severity: {risk}
Confidence: Minor={probs['Minor']}, Moderate={probs['Moderate']}, Major={probs['Major']}

The top 3 features that drove this prediction:
{shap_lines}

Write a concise clinical explanation (3-4 sentences) covering:
1. Why these two drugs may interact based on the features above
2. What the clinical risk is
3. A brief recommendation for the prescriber

Be specific, use the feature values provided, and avoid generic statements."""

    return prompt


def generate_explanation(prediction: dict) -> str:
    """
    Takes prediction dict from predict.py and returns LLM explanation string.
    """
    if 'error' in prediction:
        return f"Cannot generate explanation: {prediction['error']}"

    prompt = build_prompt(prediction)

    response = client.chat.completions.create(
        model="meta/llama-3.1-8b-instruct",
        messages=[
            {
                "role": "system",
                "content": "You are a clinical pharmacist AI. Be concise, accurate, and clinically relevant."
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        temperature=0.3,
        max_tokens=300
    )

    return response.choices[0].message.content.strip()


if __name__ == '__main__':
    # Test with a sample prediction
    sample_prediction = {
        "drug_a": "Carboplatin",
        "drug_b": "Telbivudine",
        "risk_level": "Moderate",
        "probabilities": {
            "Minor": 0.002,
            "Moderate": 0.995,
            "Major": 0.003
        },
        "top_shap_features": [
            {
                "feature": "B_ACTION_CLASS",
                "value": "5-Nitroimidazole (Antiprotozoal & Antibacterial)",
                "shap_score": 0.8097
            },
            {
                "feature": "B_liver_label",
                "value": "Unknown",
                "shap_score": 0.4974
            },
            {
                "feature": "A_ACTION_CLASS",
                "value": "Platinum compounds-Anticancer",
                "shap_score": 0.4443
            }
        ]
    }

    explanation = generate_explanation(sample_prediction)
    print("\n=== LLM Explanation ===")
    print(explanation)