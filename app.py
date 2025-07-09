from flask import Flask, render_template, request
import joblib
from collections import Counter

app = Flask(__name__)

# Load models
nb_model = joblib.load("models/spam_nb_classifier_model.pkl")
lr_model = joblib.load("models/spam_classifier_model.pkl")
svc_model = joblib.load("models/spam_svm_classifier_model.pkl")  # Make sure this path is correct

# Individual predictions
def predict_nb(message):
    return int(nb_model.predict([message])[0])

def predict_lr(message):
    return int(lr_model.predict([message])[0])

def predict_svc(message):
    return int(svc_model.predict([message])[0])

# Majority voting logic
def predict_majority(message):
    preds = [
        predict_nb(message),
        predict_lr(message),
        predict_svc(message)
    ]
    vote_count = Counter(preds)
    majority_vote = vote_count.most_common(1)[0][0]
    return majority_vote

@app.route('/', methods=['GET', 'POST'])
def index():
    result_nb = None
    result_lr = None
    result_svc = None
    result_majority = None
    message = ''

    if request.method == 'POST':
        message = request.form['message']
        result_nb = predict_nb(message)
        result_lr = predict_lr(message)
        result_svc = predict_svc(message)
        result_majority = predict_majority(message)

    return render_template(
        'index.html',
        message=message,
        result_nb=result_nb,
        result_lr=result_lr,
        result_svc=result_svc,
        result_majority=result_majority
    )

if __name__ == '__main__':
    app.run(debug=True)
