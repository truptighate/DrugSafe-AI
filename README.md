\# DrugSafe AI — Drug Interaction Risk Prediction System

&#x20;

A machine learning system that predicts the severity of drug-drug interactions and generates clinical explanations using a large language model. Built with XGBoost for prediction, SHAP for explainability, and NVIDIA NIM (LLaMA 3.1 8B) for generating human-readable clinical reasoning.

&#x20;

\---

&#x20;

\## What it does

&#x20;

You enter two drug names. The system predicts whether their interaction is Minor, Moderate, or Major, shows you the confidence behind that prediction, explains which drug properties drove the result, and generates a clinical explanation written in plain language.

&#x20;

The goal was to combine a interpretable ML model with an LLM in a way that makes the output actually useful — not just a risk score, but a reason behind it.

&#x20;

\---

&#x20;

\## How it works

&#x20;

\*\*Data pipeline\*\*

&#x20;

The project uses two datasets merged together:

&#x20;

\- \*\*MID (Medicine Information Database)\*\* — 192,807 drug records with therapeutic class, action class, chemical class, mechanism of action, and safety labels across alcohol, pregnancy, kidney, and liver categories.

\- \*\*DDInter\*\* — 222,383 labeled drug-drug interaction pairs across 8 drug categories, with severity labels (Minor, Moderate, Major).

The two datasets are joined by fuzzy-matching drug names using `rapidfuzz`. After matching and cleaning, the final training dataset contains 49,505 drug pairs, each with 16 features drawn from MID for both drugs and a severity label from DDInter.

&#x20;

\*\*Model\*\*

&#x20;

XGBoost classifier trained on the 49,505 pairs. Class weights were used to handle the label imbalance (Moderate: 75%, Major: 18%, Minor: 7%). The model achieves 86% accuracy on the held-out test set with a weighted AUC-ROC of 0.95+.

&#x20;

| Class    | Precision | Recall | F1   |

|----------|-----------|--------|------|

| Minor    | 0.67      | 0.87   | 0.76 |

| Moderate | 0.96      | 0.86   | 0.91 |

| Major    | 0.66      | 0.89   | 0.76 |

&#x20;

\*\*Explainability\*\*

&#x20;

SHAP (TreeExplainer) is run on each prediction to identify the top 3 features that drove the result. These feature values and scores are then passed as context into the LLM prompt.

&#x20;

\*\*LLM explanation\*\*

&#x20;

The SHAP output is structured into a prompt sent to LLaMA 3.1 8B via NVIDIA NIM API. The model is instructed to act as a clinical pharmacist and write a 3–4 sentence explanation covering why the interaction exists, what the clinical risk is, and what the prescriber should consider.

&#x20;

\*\*Web interface\*\*

&#x20;

FastAPI backend serving a multi-screen HTML/CSS/JS frontend. The flow is: welcome screen → drug input → results (risk + confidence + LLM explanation) → SHAP features on demand.

&#x20;

\---

&#x20;

\## Project structure

&#x20;

```

DrugInteraction\_project/

│

├── app/

│   ├── predict.py          # XGBoost + SHAP prediction logic

│   ├── explain.py          # NVIDIA NIM LLM API call

│   ├── server.py           # FastAPI backend

│   ├── gradio\_app.py       # Alternate Gradio interface

│   └── static/

│       └── index.html      # Frontend UI

│

├── data/

│   └── final\_dataset.csv   # Merged MID + DDInter dataset (49,505 rows)

│

├── models/

│   ├── xgb\_model.json      # Trained XGBoost model

│   └── label\_encoders.pkl  # Fitted LabelEncoders for categorical features

│

├── notebooks/

│   ├── 01\_EDA\_MID.ipynb

│   ├── 02\_DDInter\_Setup\_and\_Merge.ipynb

│   └── 03\_Model\_Training.ipynb

│

├── outputs/

│   ├── confusion\_matrix.png

│   ├── feature\_importance.png

│   └── shap\_summary.png

│

├── requirements.txt

└── README.md

```

&#x20;

\---

&#x20;

\## Setup

&#x20;

\*\*Requirements\*\*

&#x20;

```

Python 3.10

```

&#x20;

\*\*Install dependencies\*\*

&#x20;

```bash

pip install -r requirements.txt

pip install fastapi uvicorn

```

&#x20;

\*\*Environment variable\*\*

&#x20;

Create a `.env` file in the project root:

&#x20;

```

NVIDIA\_API\_KEY=nvapi-your-key-here

```

&#x20;

Get a free API key at \[build.nvidia.com](https://build.nvidia.com).

&#x20;

\*\*Run the app\*\*

&#x20;

```bash

uvicorn app.server:app --reload

```

&#x20;

Open `http://127.0.0.1:8000` in your browser.

&#x20;

\---

&#x20;

\## Tech stack

&#x20;

| Component | Tool |

|---|---|

| ML model | XGBoost 2.1.1 |

| Explainability | SHAP 0.46.0 |

| LLM | LLaMA 3.1 8B via NVIDIA NIM |

| Backend | FastAPI |

| Frontend | HTML / CSS / JS |

| Data processing | pandas, rapidfuzz |

| Notebooks | Google Colab |

&#x20;

\---

&#x20;

\## Features used in the model

&#x20;

For each drug in a pair, the model uses:

&#x20;

\- Therapeutic class (e.g. ANTI INFECTIVES, CARDIAC)

\- Action class (e.g. Statins, SSRIs, Platinum compounds)

\- Chemical class (e.g. Fluoroquinolone, Nucleoside analogue)

\- Habit forming (Yes/No)

\- Safety labels for alcohol, pregnancy, kidney, and liver

These 8 features per drug give 16 total input features per drug pair.

&#x20;

\---

&#x20;

\## Known limitations

&#x20;

\- Drug lookup is name-based. If a drug name doesn't exist in the training dataset, the system returns a not-found error. Brand names sometimes differ from the names used in training data.

\- The model was trained on DDInter's severity labels, which are based on known literature. Novel or rare interactions outside this dataset won't be captured.

\- LLM explanations are generated by a language model and are not verified by a pharmacist. They are meant to support, not replace, clinical judgment.



\---



\## Future scope



\- \*\*Broader drug coverage\*\* — expand the training data to include brand names, generic aliases, and international drug names so fewer lookups return not-found errors.

\- \*\*Multi-drug interactions\*\* — the current model handles pairs only. Real prescriptions often involve 5–10 drugs simultaneously. Extending to n-drug combinations is the logical next step.

\- \*\*Severity confidence calibration\*\* — the model is accurate but confidence scores are not yet calibrated. Platt scaling or isotonic regression would make the probability outputs more trustworthy for clinical use.

\- \*\*Patient-specific risk factors\*\* — incorporating patient variables like age, renal function, and CYP450 genotype would allow personalized interaction risk rather than population-level predictions.

\- \*\*Real-time drug database sync\*\* — connecting to live databases like DrugBank or RxNorm would keep the system current as new interactions are documented.

\- \*\*Clinical validation\*\* — partnering with pharmacists or hospital systems to validate predictions against real adverse event records would be a meaningful step toward actual clinical utility.

\- \*\*Mobile interface\*\* — a lightweight mobile-friendly version for use at point of care during prescribing.



\---

&#x20;

\## Disclaimer

&#x20;

This system is built for research and educational purposes. It is not a medical device and should not be used as the sole basis for clinical decisions. Always consult a licensed healthcare professional before making prescribing or medication decisions.



\---



\## Conclusion



DrugSafe AI demonstrates that a relatively small, interpretable ML model — trained on cleanly merged public datasets — can produce clinically meaningful drug interaction predictions with high accuracy. The combination of XGBoost for prediction, SHAP for explainability, and an LLM for plain-language reasoning addresses a real gap: most interaction checkers give a risk flag but no explanation of why the risk exists or what to do about it.



The 86% accuracy and 0.95+ weighted AUC-ROC on a 49,505-pair dataset are strong baselines. More importantly, the system is transparent — every prediction comes with the features that drove it, not just a number.



This was built as a research and learning project. It is not production-ready for clinical use in its current form, but it establishes a solid foundation that could be extended into something genuinely useful with broader data coverage, clinical validation, and patient-specific inputs.

