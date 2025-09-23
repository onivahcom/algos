from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional

class Service(BaseModel):
    id: str = Field(..., alias="_id")  # map _id from incoming JSON to id internally
    category: str
    amenities: List[str] = []
    thingsToKnow: List[str] = []
    offers: List[str] = []
    additionalFields: Dict[str, Any] = {}
    images: Optional[Dict[str, List[str]]] = {}  # <-- make sure this is here

    class Config:
        validate_by_name = True


class RecommendRequestData(BaseModel):
    baseService: Service
    candidates: List[Service]
