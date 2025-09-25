from fastapi import FastAPI
from services.search import search_listings
from services.spam import predict_spam
from services.recommend_service import recommend

from schemas.search_schema import SearchRequest
from schemas.spam_schema import RequestData
from schemas.recommend_schema import RecommendRequestData

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


app = FastAPI()


@app.get("/")
def home():
    return {"message": "Python service is running successfully🚀"}


@app.post("/search")
def search(data: SearchRequest):
    listings_data = [listing.model_dump() for listing in data.listings]
    ranked = search_listings(data.query, listings_data)
    return {"ranked_listings": ranked}


@app.post("/predict")
def predict(data: RequestData):
    return predict_spam(data.text)

@app.post("/recommend")
def recommend_endpoint(data: RecommendRequestData):
    return recommend(data.baseService, data.candidates)

