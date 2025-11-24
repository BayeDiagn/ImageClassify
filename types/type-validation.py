from pydantic import BaseModel
from typing import List, Optional




class PredictionResponse(BaseModel):
    """Réponse de prédiction"""
    model_name: str
    predicted_class: str
    probability: float


class ErrorResponse(BaseModel):
    """Réponse d'erreur"""
    error: str
    detail: Optional[str] = None