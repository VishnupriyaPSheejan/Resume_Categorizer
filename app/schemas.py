# # app/schemas.py
# from pydantic import BaseModel
# from typing import List, Optional

# class ResumeRequest(BaseModel):
#     text: str  # Raw resume text from user

# class ResumeResponse(BaseModel):
#     predicted_department: str
#     predicted_role: str
#     confidence: float
#     top_alternatives: Optional[List[str]] = None
# app/schemas.py
from pydantic import BaseModel
from typing import List, Optional

class ResumeRequest(BaseModel):
    text: str  # Raw resume text from user

class ResumeResponse(BaseModel):
    predicted_department: str
    predicted_role: str
    confidence: float
    top_alternatives: Optional[List[str]] = None
