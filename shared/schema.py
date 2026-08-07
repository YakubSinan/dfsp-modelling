from dataclasses import dataclass
from typing import List, Optional


@dataclass
class LesionPrediction:
    x: float
    y: float
    confidence: float
    lesion_id: Optional[str] = None


@dataclass
class ModelOutput:
    image_id: str
    view: str
    predictions: List[LesionPrediction]
    patient_id: Optional[str] = None
    