from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def search_listings(
    query, listings, threshold=0.1,
    weight_text=0.6, weight_popularity=0.2, weight_reviews=0.15, weight_distance=0.05
):
    # Prepare documents by combining name, description, and boosted locations
    documents = []
    for item in listings:
        name = item.get('name', '') or ''
        description = item.get('description', '') or ''
        locations = item.get('locations', [])  # list of strings

        # Boost locations by repeating location terms multiple times (e.g. 5 times)
        locations_boosted = " ".join(locations * 5) if locations else ""
        combined_text = f"{name} {description} {locations_boosted}".strip()
        documents.append(combined_text.lower())  # lowercase for case-insensitive matching

    if not any(documents):
        return []

    # Vectorize documents and query
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(documents)
    query_vector = vectorizer.transform([query.lower()])  # lowercase query

    # Calculate similarity scores
    tfidf_scores = cosine_similarity(query_vector, tfidf_matrix)[0]

    scored_listings = []
    for i, score in enumerate(tfidf_scores):
        # print(f"Listing {i} score: {score:.4f}, text: {documents[i]}") 
        
        if score >= threshold:
            listing = listings[i]
            popularity = listing.get("popularity", 0)
            reviews = listing.get("reviews", 0)
            distance = listing.get("distance", 0)

            final_score = (
                weight_text * score +
                weight_popularity * popularity +
                weight_reviews * reviews -
                weight_distance * distance
            )

            scored_listings.append({
                **listing,
                "tfidf_score": float(score),
                "final_score": final_score
            })

    # Sort by final combined score descending
    ranked = sorted(scored_listings, key=lambda x: x["final_score"], reverse=True)
    return ranked
