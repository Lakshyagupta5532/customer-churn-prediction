# 📊 Customer Churn Prediction

An ML-powered web application that predicts whether a telecom customer is likely to churn (leave) based on their usage patterns, contract type, and billing information.

## 🧩 The Problem

Telecom companies lose significant revenue when customers switch to competitors. Identifying at-risk customers **before** they leave allows companies to take proactive retention actions — like offering discounts or improved plans.

## 💡 How It Works

1. Customer data is collected (contract type, tenure, services, billing)
2. Data is preprocessed — categorical variables encoded, features scaled
3. A **Logistic Regression** model predicts churn probability
4. Results show churn risk level with actionable business recommendations

## 📊 Model Performance

| Model | Accuracy |
|-------|----------|
| Logistic Regression | 78.75% |
| Random Forest | 78.68% |
| Decision Tree | 72.64% |

**Best Model:** Logistic Regression (78.75% accuracy, F1-Score: 0.56 for churn class)

## ✨ Features

- 👤 Enter customer details (personal info, services, billing)
- 🔮 Instant churn prediction with probability score
- 🟢🟡🔴 Risk level categorization (Low / Medium / Critical)
- 💡 Actionable retention recommendations

## 🛠️ Tech Stack

- **Python**
- **Pandas, NumPy** — data preprocessing
- **Scikit-learn** — Logistic Regression, Random Forest, Decision Tree
- **Matplotlib, Seaborn** — visualization
- **Streamlit** — web application

## 📊 Dataset

[Telco Customer Churn Dataset](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) — 7,032 customers with 20 features.

## 🚀 Live Demo

🔗 [Not deployed yet]

## 🖥️ Run Locally

```bash
git clone https://github.com/Lakshyagupta5532/customer-churn-prediction.git
cd customer-churn-prediction
pip install -r requirements.txt
streamlit run app.py
```

## 🧠 What I Learned

- Handling imbalanced datasets and understanding why accuracy alone is misleading
- Difference between Precision, Recall, and F1-Score — and when each matters
- One-Hot Encoding for categorical variables
- Comparing multiple ML models and selecting the best one
- End-to-end ML pipeline from raw data to deployed web app

## 🔮 Future Improvements

- Add batch prediction (CSV upload for multiple customers at once)
- Implement SMOTE for handling class imbalance
- Add feature importance visualization
- Experiment with XGBoost for better performance
