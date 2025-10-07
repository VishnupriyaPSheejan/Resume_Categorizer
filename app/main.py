# from fastapi import FastAPI
# from app.schemas import ResumeRequest, ResumeResponse
# from app.inference import categorize_resume

# app = FastAPI(title="Resume Categorization & Talent Analytics System")

# @app.get("/")
# def health_check():
#     return {"status": "running"}

# @app.post("/categorize", response_model=ResumeResponse)
# def categorize(resume: ResumeRequest):
#     result = categorize_resume(resume.text)
#     return result
# app/main.py
from fastapi import FastAPI
from app.schemas import ResumeRequest, ResumeResponse
from app.inference import categorize_resume

app = FastAPI(title="Resume Categorization & Talent Analytics System")

@app.get("/")
def health_check():
    """Simple health check endpoint."""
    return {"status": "running"}

@app.post("/categorize", response_model=ResumeResponse)
def categorize(resume: ResumeRequest):
    """
    Accepts resume text as input and returns department/role predictions.
    """
    result = categorize_resume(resume.text)
    return result
