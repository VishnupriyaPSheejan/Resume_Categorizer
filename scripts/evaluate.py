# scripts/evaluate.py
import pandas as pd
import torch
import os
import pickle
from sklearn.metrics import accuracy_score, classification_report
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from app.inference import categorize_resume

# 1. Load test dataset
df = pd.read_csv("data/jobss.csv")

df['text'] = df['Job Title'] + " " + df['Key Skills'].fillna("") + " " + df['Role Category'].fillna("")
df = df[df['Functional Area'].notnull()]
true_labels = df['Functional Area'].tolist()

# 2. Load fine-tuned model + tokenizer + label encoder
model = AutoModelForSequenceClassification.from_pretrained("fine_tuned_resume_bert")
tokenizer = AutoTokenizer.from_pretrained("fine_tuned_resume_bert")
with open("fine_tuned_resume_bert/label_encoder.pkl", "rb") as f:
    le = pickle.load(f)

# Override categorize_resume to use fine-tuned model and real labels
def categorize_resume_real(resume_text: str):
    inputs = tokenizer(resume_text, return_tensors="pt", truncation=True, padding=True, max_length=128)
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
    confidence, predicted_idx = torch.max(probs, dim=-1)
    confidence = confidence.item()
    predicted_idx = predicted_idx.item()
    predicted_label = le.inverse_transform([predicted_idx])[0]
    return predicted_label

# 3. Predict all test examples
predicted_labels = []
for t in df['text'].tolist():
    predicted_labels.append(categorize_resume_real(t))

# 4. Evaluation
print("Accuracy:", accuracy_score(true_labels, predicted_labels))
print("\nClassification Report:\n", classification_report(true_labels, predicted_labels))
