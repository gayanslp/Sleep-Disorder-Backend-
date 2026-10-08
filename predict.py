import os
import sys
import json
import joblib
import pandas as pd

# Locate the trained model
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, 'models', 'sleep_disorder_lightgbm_model.joblib')

if not os.path.exists(MODEL_PATH):
    # Fallback to parent directory if not in models/
    MODEL_PATH = os.path.join(os.path.dirname(BASE_DIR), 'sleep_disorder_lightgbm_model.joblib')

if not os.path.exists(MODEL_PATH):
    sys.stderr.write(f"Error: Model file not found at {MODEL_PATH}\n")
    sys.exit(1)

try:
    model = joblib.load(MODEL_PATH)
except Exception as e:
    sys.stderr.write(f"Error loading model: {str(e)}\n")
    sys.exit(1)

CLASS_MAP = {
    0: 'Insomnia',
    1: 'None',
    2: 'Sleep Apnea'
}

OCCUPATIONS = [
    'Occupation_Accountant',
    'Occupation_Doctor',
    'Occupation_Engineer',
    'Occupation_Lawyer',
    'Occupation_Manager',
    'Occupation_Nurse',
    'Occupation_Sales_Representative',
    'Occupation_Salesperson',
    'Occupation_Scientist',
    'Occupation_Software_Engineer',
    'Occupation_Teacher'
]

def preprocess_input(data):
    # 1. Gender: Male -> 1, Female -> 0
    gender_str = str(data.get('gender', '')).strip().lower()
    gender = 1 if gender_str == 'male' else 0

    # 2. BMI Category: Normal -> 0, Obese -> 1, Overweight -> 2
    bmi_str = str(data.get('bmi_category', '')).strip().lower()
    if 'obese' in bmi_str:
        bmi_val = 1
    elif 'overweight' in bmi_str:
        bmi_val = 2
    else:
        bmi_val = 0

    # 3. Blood Pressure (e.g. "120/80")
    bp = str(data.get('blood_pressure', '120/80')).replace(' ', '')
    try:
        parts = bp.split('/')
        systolic = float(parts[0])
        diastolic = float(parts[1])
    except Exception:
        systolic = 120.0
        diastolic = 80.0

    # 4. Occupation one-hot encoding
    user_occ = str(data.get('occupation', '')).strip().lower().replace(' ', '_')
    occ_values = {}
    for occ_col in OCCUPATIONS:
        canonical_name = occ_col.replace('Occupation_', '').lower()
        occ_values[occ_col] = 1 if user_occ == canonical_name else 0

    row = {
        'Gender': gender,
        'Age': float(data.get('age', 30)),
        'Sleep_Duration': float(data.get('sleep_duration', 7.0)),
        'Quality_of_Sleep': float(data.get('quality_of_sleep', 7)),
        'Physical_Activity_Level': float(data.get('physical_activity_level', 40)),
        'Stress_Level': float(data.get('stress_level', 5)),
        'BMI_Category': bmi_val,
        'Heart_Rate': float(data.get('heart_rate', 70)),
        'Daily_Steps': float(data.get('daily_steps', 6000)),
        'Systolic_BP': systolic,
        'Diastolic_BP': diastolic,
        **occ_values
    }

    features = getattr(model, 'feature_name_', list(row.keys()))
    df = pd.DataFrame([row])[features]
    return df

def main():
    try:
        # Read JSON input either from argument or stdin
        if len(sys.argv) > 1:
            raw_input = sys.argv[1]
        else:
            raw_input = sys.stdin.read().strip()

        if not raw_input:
            sys.stderr.write("Error: No input data provided.\n")
            sys.exit(1)

        data = json.loads(raw_input)
        df = preprocess_input(data)

        preds = model.predict(df)
        probas = model.predict_proba(df)[0]

        pred_idx = int(preds[0])
        label = CLASS_MAP.get(pred_idx, 'Unknown')
        confidence = float(probas[pred_idx])

        result = {
            'prediction': label,
            'confidence': round(confidence, 4),
            'probabilities': {
                CLASS_MAP[i]: round(float(p), 4) for i, p in enumerate(probas)
            }
        }

        print(json.dumps(result))
    except Exception as e:
        sys.stderr.write(f"Inference error: {str(e)}\n")
        sys.exit(1)

if __name__ == '__main__':
    main()
