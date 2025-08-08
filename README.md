# Spam Detection Web App

## 📌 Overview
This is a **Flask-based web application** that detects whether a given message is **spam** or **not spam** using three different machine learning models:

1. **Naive Bayes Classifier** (`spam_nb_classifier_model.pkl`)
2. **Logistic Regression Classifier** (`spam_classifier_model.pkl`)
3. **Support Vector Machine (SVM) Classifier** (`spam_svm_classifier_model.pkl`)

The app predicts results from all three models and also performs **majority voting** to give a final verdict.

---

## 🚀 Features
- Accepts a text message as input
- Runs the message through **three trained models**
- Displays:
  - Naive Bayes prediction
  - Logistic Regression prediction
  - SVM prediction
  - Majority voting result
- Simple and user-friendly web interface

---

## 🛠️ Requirements
- Python 3.7+
- Flask
- scikit-learn
- joblib

You can install the required dependencies by running:
```bash
pip install flask scikit-learn joblib


project/
│
├── app.py                        
├── models/
│   ├── spam_nb_classifier_model.pkl
│   ├── spam_classifier_model.pkl
│   └── spam_svm_classifier_model.pkl
├── templates/
│   └── index.html                 
└── README.md                      


▶️ How to Run
Clone this repository or copy the project files into a folder.

Place the trained models in the models/ directory.

Run the Flask app:

python app.py
