❤️ Heart Disease Prediction Using Machine Learning
📌 Project Overview

Heart disease remains one of the leading causes of death worldwide. Early detection plays a crucial role in prevention and effective treatment.
This project builds an end-to-end Machine Learning pipeline to predict the likelihood of heart disease using clinical and demographic data, and deploys the model as an interactive Streamlit web application.

The solution covers:

Data preprocessing and feature engineering

Model training and evaluation

Model persistence (joblib)

Interactive prediction UI using Streamlit

🎯 Objectives

Predict whether a patient is at risk of heart disease

Provide probability-based confidence scores

Build a deployable and reproducible ML system

Demonstrate production-ready ML skills for portfolio and recruiters

📊 Dataset Description

The dataset consists of clinical records of 303 patients, commonly used in heart disease research.

Features Used (13)
Feature	Description
age	Age of patient
sex	Gender (1 = Male, 0 = Female)
cp	Chest pain type
trestbps	Resting blood pressure
chol	Serum cholesterol
fbs	Fasting blood sugar > 120 mg/dl
restecg	Resting ECG results
thalach	Maximum heart rate achieved
exang	Exercise-induced angina
oldpeak	ST depression
slope	Slope of peak exercise ST
ca	Number of major vessels
thal	Thalassemia
Target Variable

0 → Healthy

1 → Heart Disease

🧠 Machine Learning Approach
Model Used

Logistic Regression

Interpretable

Well-suited for binary classification

Strong baseline for medical datasets

Preprocessing

Feature scaling using StandardScaler

Feature alignment enforced during inference

Train/Test split for unbiased evaluation

📈 Model Evaluation
Metrics Used

Accuracy

Cross-validation (5-fold)

Classification Report (Precision, Recall, F1-score)

Confusion Matrix

Probability scores (predict_proba)

Sample Performance

Accuracy: ~85%

Balanced performance between precision and recall

Suitable for early risk screening

Confusion Matrix Visualization

The model performance is visualized using a confusion matrix heatmap to clearly show:

True Positives

True Negatives

False Positives

False Negatives

🚀 Deployment (Streamlit App)

An interactive Streamlit web application allows users to:

Input patient health metrics

Get real-time predictions

View confidence level and risk category

See probability distribution (Healthy vs Disease)

Key UI Features

Clean, responsive layout

Risk categorization: Low / Medium / High

Probability bar chart using Plotly

Medical disclaimer for ethical use

🗂️ Project Structure
heart-disease-prediction/
│
├── app.py                     # Streamlit application
├── heart_disease_model.pkl    # Saved model + feature names
├── scaler.pkl                 # Trained scaler
├── notebook.ipynb             # EDA + training notebook
├── confusion_matrix.png       # Model evaluation visual
├── requirements.txt           # Dependencies
└── README.md                  # Project documentation

💾 Model Persistence

The trained artifacts are saved using joblib:

model_data = {
    "model": model,
    "feature_names": X.columns.tolist()
}
joblib.dump(model_data, "heart_disease_model.pkl")


⚠️ Disclaimer

This project is for educational and demonstration purposes only.
It is not a substitute for professional medical diagnosis or treatment.

🧑‍💻 Skills Demonstrated

Data preprocessing & feature engineering

Machine Learning model development

Model evaluation & validation

Model deployment (Streamlit)

Production-ready ML pipelines

GitHub documentation & presentation

🌍 Future Improvements

Try advanced models (Random Forest, XGBoost)

Hyperparameter tuning

ROC-AUC optimization

Model explainability (SHAP / LIME)

Cloud deployment (Streamlit Cloud / Docker)

## Meanings of Medical Terms with High Impact on the Research

# These four clinical terms strongly induce diseases on patients evaluated

1. Chest Pain (cp – Chest Pain Type)

This refers to discomfort or pain felt in the chest, often related to reduced blood flow to the heart muscle.

Common categories include:

Typical angina – Classic heart-related chest pain, usually triggered by exertion or stress.

Atypical angina – Chest pain with some, but not all, features of classic angina.

Non-anginal pain – Chest pain not related to the heart (e.g., muscle or digestive causes).

Asymptomatic – No chest pain present.

2. restecg (Resting Electrocardiogram Results)

This describes the electrical activity of the heart measured while the patient is at rest.

Common values:

Normal – No visible abnormalities.

ST-T wave abnormality – May indicate heart muscle ischemia or electrolyte imbalance.

Left ventricular hypertrophy – Thickening of the heart’s left ventricle, often due to high blood pressure.

3. thalach (Maximum Heart Rate Achieved)

This represents the highest heart rate (beats per minute) reached during exercise or stress testing.

Lower values may indicate poor cardiovascular fitness or heart disease.

Higher values generally suggest better heart performance, depending on age.

4. slope (Slope of the Peak Exercise ST Segment)

This refers to the pattern of the ST segment on an ECG during peak exercise, which helps assess heart stress.

Types:

Upsloping – Usually normal or low risk.

Flat – May indicate moderate heart disease.

Downsloping – Strongly associated with myocardial ischemia (reduced blood flow to the heart).

📬 Contact

Author: Julius Duru
Role: Aspiring Data Scientist | Presales Engineer
Focus: Machine Learning, Data Science, AI Applications