import joblib
import shap
import numpy as np
import pandas as pd
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

from xgboost import XGBClassifier
model = XGBClassifier()
model.load_model(os.path.join(BASE_DIR, 'models', 'xgb_model.json'))
encoders  = joblib.load(os.path.join(BASE_DIR, 'models', 'label_encoders.pkl'))
df        = pd.read_csv(os.path.join(BASE_DIR, 'data', 'final_dataset.csv'))

explainer = shap.TreeExplainer(model)

LABEL_MAP = {0: 'Minor', 1: 'Moderate', 2: 'Major'}

CAT_COLS = [
    'A_THERAPEUTIC_CLASS', 'A_ACTION_CLASS', 'A_CHEMICAL_CLASS',
    'A_HABIT_FORMING', 'A_alcohol_label', 'A_pregnancy_label',
    'A_kidney_label', 'A_liver_label',
    'B_THERAPEUTIC_CLASS', 'B_ACTION_CLASS', 'B_CHEMICAL_CLASS',
    'B_HABIT_FORMING', 'B_alcohol_label', 'B_pregnancy_label',
    'B_kidney_label', 'B_liver_label'
]

def get_drug_features(drug_name: str, prefix: str) -> dict:
    """Look up drug features from dataset by name."""
    drug_name = drug_name.strip().lower()

    # Search Drug_A column first, then Drug_B
    row = df[df['Drug_A'].str.lower() == drug_name]
    if row.empty:
        row = df[df['Drug_B'].str.lower() == drug_name]
    if row.empty:
        return None

    row = row.iloc[0]
    features = {}
    for col in CAT_COLS:
        if col.startswith('A_'):
            original_col = col[2:]
            features[f'{prefix}_{original_col}'] = row[f'A_{original_col}']
        else:
            original_col = col[2:]
            features[f'{prefix}_{original_col}'] = row[f'B_{original_col}']
    return features


def predict_interaction(drug_a: str, drug_b: str) -> dict:
    """
    Main prediction function.
    Returns risk level, probabilities, and top SHAP features.
    """
    # Get features for both drugs
    features_a = get_drug_features(drug_a, 'A')
    features_b = get_drug_features(drug_b, 'B')

    if features_a is None:
        return {'error': f'Drug "{drug_a}" not found in database.'}
    if features_b is None:
        return {'error': f'Drug "{drug_b}" not found in database.'}

    # Build input row
    input_data = {}
    for col in CAT_COLS:
        prefix = col[0]
        original = col[2:]
        input_data[col] = features_a[f'A_{original}'] if prefix == 'A' \
                          else features_b[f'B_{original}']

    input_df = pd.DataFrame([input_data])

    # Encode
    for col in CAT_COLS:
        le = encoders[col]
        val = str(input_df[col].values[0])
        if val in le.classes_:
            input_df[col] = le.transform([val])[0]
        else:
            input_df[col] = 0

    # Predict
    pred_label   = int(model.predict(input_df)[0])
    pred_proba   = model.predict_proba(input_df)[0]
    risk_level   = LABEL_MAP[pred_label]

    # SHAP
    shap_vals    = explainer.shap_values(input_df)
    shap_for_class = shap_vals[0][:, pred_label]
    shap_series  = pd.Series(shap_for_class, index=CAT_COLS)
    top_features = shap_series.abs().sort_values(ascending=False).head(3)

    top_shap = []
    for feat in top_features.index:
        raw_val     = input_df[feat].values[0]
        decoded_val = encoders[feat].inverse_transform([raw_val])[0]
        top_shap.append({
            'feature'     : feat,
            'value'       : decoded_val,
            'shap_score'  : round(float(shap_series[feat]), 4)
        })

    return {
        'drug_a'      : drug_a,
        'drug_b'      : drug_b,
        'risk_level'  : risk_level,
        'probabilities': {
            'Minor'    : round(float(pred_proba[0]), 3),
            'Moderate' : round(float(pred_proba[1]), 3),
            'Major'    : round(float(pred_proba[2]), 3)
        },
        'top_shap_features': top_shap
    }


if __name__ == '__main__':
    result = predict_interaction('Carboplatin', 'Telbivudine')
    import json
    print(json.dumps(result, indent=2))