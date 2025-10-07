# # app/inference.py
# from transformers import AutoTokenizer, AutoModelForSequenceClassification
# import torch
# import os

# # Load model only once (better for performance)
# MODEL_NAME = os.getenv("MODEL_NAME", "bert-base-uncased")  # Replace with your fine-tuned model path
# tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
# model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)

# # Define department/role labels based on your fine-tuned model
# DEPARTMENTS = ["Software Development", "Data Science", "QA/Testing", "DevOps", "HR", "Finance"]
# ROLES = ["Backend Engineer", "Frontend Engineer", "ML Engineer", "QA Engineer", "HR Executive", "Business Analyst"]

# def categorize_resume(resume_text: str):
#     # Tokenize input
#     inputs = tokenizer(resume_text, return_tensors="pt", truncation=True, padding=True)

#     # Get model predictions
#     with torch.no_grad():
#         outputs = model(**inputs)
#         probs = torch.nn.functional.softmax(outputs.logits, dim=-1)

#     confidence, predicted_idx = torch.max(probs, dim=-1)
#     confidence = confidence.item()
#     predicted_idx = predicted_idx.item()

#     # Map predicted index to a department/role (placeholder mapping)
#     predicted_department = DEPARTMENTS[predicted_idx % len(DEPARTMENTS)]
#     predicted_role = ROLES[predicted_idx % len(ROLES)]

#     # Return response
#     return {
#         "predicted_department": predicted_department,
#         "predicted_role": predicted_role,
#         "confidence": confidence,
#         "top_alternatives": [DEPARTMENTS[i] for i in probs.topk(3).indices.tolist()[0]]
#     }

# app/inference.py
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import os

# Load model only once (cached in memory for performance)
MODEL_NAME = os.getenv("MODEL_NAME", "bert-base-uncased")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)

# Sample department/role labels — adjust based on your dataset
DEPARTMENTS = ["Software Development", "Data Science", "QA/Testing", "DevOps", "HR", "Finance"]
ROLES = ["Backend Engineer", "Frontend Engineer", "ML Engineer", "QA Engineer", "HR Executive", "Business Analyst"]

# # app/inference.py
# def categorize_resume(resume_text: str):
#     inputs = tokenizer(resume_text, return_tensors="pt", truncation=True, padding=True)

#     with torch.no_grad():
#         outputs = model(**inputs)
#         probs = torch.nn.functional.softmax(outputs.logits, dim=-1)

#     confidence, predicted_idx = torch.max(probs, dim=-1)
#     confidence = confidence.item()
#     predicted_idx = predicted_idx.item()

#     num_classes = probs.shape[-1]  # Get number of model outputs
#     top_k = min(3, num_classes)    # Avoid out-of-range error
#     top_indices = probs.topk(top_k).indices.tolist()[0]

#     predicted_department = DEPARTMENTS[predicted_idx % len(DEPARTMENTS)]
#     predicted_role = ROLES[predicted_idx % len(ROLES)]

#     return {
#         "predicted_department": predicted_department,
#         "predicted_role": predicted_role,
#         "confidence": confidence,
#         "top_alternatives": [DEPARTMENTS[i % len(DEPARTMENTS)] for i in top_indices]
    # }
# scripts/inference.py
# scripts/inference.py
# import torch
# import os
# from transformers import BertTokenizer, BertForSequenceClassification

# # Load your fine-tuned model
# BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# # MODEL_PATH = os.path.join(BASE_DIR, "..","fine_tuned_resume_bert")
# MODEL_PATH = MODEL_PATH = r"C:\Users\vishnupriya.s\project1\scripts\fine_tuned_resume_bert"

# tokenizer = BertTokenizer.from_pretrained(MODEL_PATH)
# model = BertForSequenceClassification.from_pretrained(MODEL_PATH)
# model.eval()

# def predict_job_role(resume_text):
#     # Tokenize input
#     inputs = tokenizer(resume_text, return_tensors="pt", truncation=True, padding=True)

#     # Get model prediction
#     with torch.no_grad():
#         outputs = model(**inputs)
#         predicted_class_id = torch.argmax(outputs.logits, dim=1).item()

#     return predicted_class_id

# if __name__ == "__main__":
#     # Example test resume
#     sample_resume = "Experienced data scientist skilled in Python, machine learning, and big data analytics."
#     prediction = predict_job_role(sample_resume)
#     decoded_label = label_encoder.inverse_transform([prediction])[0]

#     print(f"Predicted class: {prediction}")
#     print(f"Predicted label: {decoded_label}")














#the following works perfectly
import torch
import os
import joblib
from transformers import BertTokenizer, BertForSequenceClassification

# Load your fine-tuned model
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = r"C:\Users\vishnupriya.s\project1\scripts\fine_tuned_resume_bert"

tokenizer = BertTokenizer.from_pretrained(MODEL_PATH)
model = BertForSequenceClassification.from_pretrained(MODEL_PATH)
model.eval()

# ✅ Load label encoder here
label_encoder = joblib.load(os.path.join(MODEL_PATH, "label_encoder.pkl"))

def predict_job_role(resume_text):
    # Tokenize input
    inputs = tokenizer(resume_text, return_tensors="pt", truncation=True, padding=True)

    # Get model prediction
    with torch.no_grad():
        outputs = model(**inputs)
        predicted_class_id = torch.argmax(outputs.logits, dim=1).item()

    return predicted_class_id

if __name__ == "__main__":
    # Example test resume
    sample_resume = "Experienced data scientist skilled in Python, machine learning, and big data analytics."
    prediction = predict_job_role(sample_resume)

    # ✅ Decode class to human-readable label
    decoded_label = label_encoder.inverse_transform([prediction])[0]

    print(f"Predicted class: {prediction}")
    print(f"Predicted label: {decoded_label}")


# import pandas as pd
# import torch
# import joblib
# from transformers import BertTokenizer, BertForSequenceClassification
# from sklearn.metrics import accuracy_score, classification_report

# # Paths
# MODEL_PATH = r"C:\Users\vishnupriya.s\project1\scripts\fine_tuned_resume_bert"
# TEST_DATA_PATH = TEST_DATA_PATH = r"C:\Users\vishnupriya.s\project1\data\jobss.csv"
#   # your test dataset

# # Load model and tokenizer
# tokenizer = BertTokenizer.from_pretrained(MODEL_PATH)
# model = BertForSequenceClassification.from_pretrained(MODEL_PATH)
# model.eval()

# # Load label encoder
# label_encoder = joblib.load(f"{MODEL_PATH}/label_encoder.pkl")

# # Load test dataset
# df = pd.read_csv(TEST_DATA_PATH)
# texts = df["Job Title"].tolist()  # or the column that contains resume/job description
# true_labels = df["Functional Area"].tolist()  # actual labels

# predicted_labels = []

# for text in texts:
#     # Tokenize
#     inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
#     with torch.no_grad():
#         outputs = model(**inputs)
#         pred_class = torch.argmax(outputs.logits, dim=1).item()
    
#     # Decode label
#     pred_label = label_encoder.inverse_transform([pred_class])[0]
#     predicted_labels.append(pred_label)

# # Calculate metrics
# accuracy = accuracy_score(true_labels, predicted_labels)
# report = classification_report(true_labels, predicted_labels)

# print(f"Accuracy: {accuracy:.4f}")
# print("Classification Report:")
# print(report)
