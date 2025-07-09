import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import joblib
import sys

# 1. Load CSV into DataFrame
df = pd.read_csv('../spam.csv', encoding='latin-1', low_memory=False)
print("Loaded columns:", df.columns.tolist())

# 2. Select message text and label columns
if 'text' in df.columns and 'target' in df.columns:
    df = df[['text', 'target']]
elif 'v2' in df.columns and 'v1' in df.columns:
    df = df[['v2', 'v1']].rename(columns={'v2': 'text', 'v1': 'target'})
elif df.shape[1] >= 2:
    df = df.iloc[:, [1, 2]].copy()
    df.columns = ['text', 'target']
else:
    print("Error: could not find appropriate text/label columns.", df.columns.tolist())
    sys.exit(1)

# 3. Drop missing entries
df = df.dropna(subset=['text', 'target'])
print(f"After dropna: {len(df)} rows.")

# 4. Normalize labels to integers
df['target'] = pd.to_numeric(df['target'], errors='ignore')

def parse_label(val):
    if isinstance(val, (int, float)):
        return int(val)
    s = str(val).strip().lower()
    if s in ('spam', '1', 'true', 'yes'):
        return 1
    if s in ('ham', '0', 'false', 'no'):
        return 0
    return None

df['target'] = df['target'].apply(parse_label)
df = df.dropna(subset=['target'])
df['target'] = df['target'].astype(int)
print(f"After label parsing: {len(df)} rows.")

# 5. Prepare data
X = df['text']
y = df['target']

# 6. Split train/test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 7. Build pipeline with SVM (LinearSVC)
pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(stop_words='english', max_df=0.9, min_df=5)),
    ('clf', LinearSVC(C=1.0, max_iter=1000, random_state=42))
])

# 8. Train
print("Training SVM model...")
pipeline.fit(X_train, y_train)

# 9. Evaluate
print("Evaluating on test set...")
y_pred = pipeline.predict(X_test)
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

# 10. Save model
model_filepath = 'spam_svm_classifier_model.pkl'
joblib.dump(pipeline, model_filepath)
print(f"Model saved to {model_filepath}")

# 11. Prediction helper
def predict_spam(message: str) -> int:
    """Return 1 for spam, 0 for ham."""
    try:
        pipeline
    except NameError:
        pipeline = joblib.load(model_filepath)
    return int(pipeline.predict([message])[0])

if __name__ == '__main__':
    sample = "Congratulations, you're selected for a free iPhone!"
    result = predict_spam(sample)
    print(f"Message: {sample}\nPrediction: {result} ({'SPAM' if result else 'HAM'})")
