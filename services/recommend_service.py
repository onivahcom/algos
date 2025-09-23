from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from schemas.recommend_schema import Service

def build_text(service: Service) -> str:

    """Convert service attributes into a single text string for similarity."""
    parts = []

    # Base category
    if service.category:
        parts.append(service.category)

    # Include description
    description = service.additionalFields.get("description")
    if description:
        parts.append(description)

    # Include amenities, thingsToKnow, and offers from additionalFields
    for field in ["amenities", "thingsToKnow", "offers"]:
        parts.extend(service.additionalFields.get(field, []))

    # Include other additionalFields (excluding ones already added)
    skip_fields = {"description", "amenities", "thingsToKnow", "offers"}
    for k, v in (service.additionalFields or {}).items():
        if k in skip_fields:
            continue
        if isinstance(v, list):
            for item in v:
                if isinstance(item, dict):
                    parts.extend(str(x) for x in item.values())
                else:
                    parts.append(str(item))
        elif isinstance(v, dict):
            parts.extend(str(x) for x in v.values())
        else:
            parts.append(str(v))

    return " ".join(parts)

#  # Include other additionalFields (excluding ones already added)
#     skip_fields = {"description", "amenities", "thingsToKnow", "offers"}
#     for k, v in (service.additionalFields or {}).items():
#         if k in skip_fields:
#             continue
#         if isinstance(v, list):
#             for item in v:
#                 if isinstance(item, dict):
#                     for key, val in item.items():
#                         parts.append(str(val))
#                         print(f"Added {k}.{key}:", val)
#                 else:
#                     parts.append(str(item))
#                     print(f"Added {k} list item:", item)
#         elif isinstance(v, dict):
#             for key, val in v.items():
#                 parts.append(str(val))
#                 print(f"Added {k}.{key}:", val)
#         else:
#             parts.append(str(v))
#             print(f"Added {k}:", v)

#     print("--- End of build_text ---\n")
#     return " ".join(parts)

def recommend(base_service: Service, candidates: list[Service]) -> list[dict]:
    MIN_SIMILARITY = 0.4  # threshold for similarity

    # 1. Build text corpus
    base_text = build_text(base_service)
    candidate_texts = [build_text(s) for s in candidates]

    # 2. Vectorize
    vectorizer = TfidfVectorizer(stop_words="english")
    vectors = vectorizer.fit_transform([base_text] + candidate_texts)

    # 3. Compute similarity
    similarities = cosine_similarity(vectors[0:1], vectors[1:]).flatten()

    # 4. Rank candidates
    ranked = sorted(
        zip(candidates, similarities),
        key=lambda x: x[1],
        reverse=True
    )

   # 5. Prepare response
    response = []

    for s, score in ranked:
         # Skip low similarity
        # if score < MIN_SIMILARITY:
        #     continue
        cover_image = None
        if getattr(s, "images", None):
            cover_image_list = s.images.get("CoverImage", [])
            if cover_image_list:
                cover_image = cover_image_list[0]  # first image

        resp_item = {
            "id": s.id,
            "similarity": float(score),
            "category": s.category,
            "coverImage": cover_image,
            "description": s.additionalFields.get("description"),
            "businessName": s.additionalFields.get("businessName"),
            "locations":s.additionalFields.get("availableLocations")
        }
        response.append(resp_item)

    # 6. Return response
    return response
