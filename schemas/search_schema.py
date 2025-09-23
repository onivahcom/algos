from pydantic import BaseModel
from typing import List, Optional


class Listing(BaseModel):
    id: Optional[str] = None
    name: str
    description: Optional[str] = None
    popularity: Optional[int] = 0
    reviews: Optional[int] = 0
    distance: Optional[float] = 0.0
    locations: Optional[List[str]] = []   


class SearchRequest(BaseModel):
    query: str
    listings: List[Listing]
