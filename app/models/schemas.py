from pydantic import BaseModel, Field
from typing import List

# --- Models for the API Request and Response ---

class HackRxRequest(BaseModel):
    documents: str = Field(..., description="URL to the PDF document.")
    questions: List[str] = Field(..., description="List of questions to answer based on the document.")

class HackRxResponse(BaseModel):
    # The response will now contain a list of simple string answers.
    answers: List[str]