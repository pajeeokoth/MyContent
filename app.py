import os
import pickle
import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware # to allow cross-origin requests
from typing import List
from surprise import Dataset, Reader, SVD, accuracy
from surprise.model_selection import train_test_split, cross_validate
import logging
from fastapi.responses import JSONResponse

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# -----------------------
# Load and combine CSVs
# -----------------------
base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "clicks_ext", "clicks"))
directory = base_dir
csv_files = [f for f in os.listdir(directory) if f.endswith(".csv")]
if not csv_files:
    raise FileNotFoundError(f"No CSV files found in {directory}")
df = pd.concat(
    [pd.read_csv(os.path.join(directory, file)) for file in csv_files],
    ignore_index=True
)

# -----------------------
# Load model (from local)
# -----------------------

def load_model_auto():
    # candidate local filenames (in order)
    candidates = [
        "svd_algo_retrained.pkl",
        "svd_algo.pkl",
        "svd_model.pkl",
        "svd_algo_retrained.pkl"  # keep common names
    ]
    base_dir = os.path.dirname(__file__)
    for name in candidates:
        path = os.path.abspath(os.path.join(base_dir, name))
        if os.path.exists(path):
            try:
                with open(path, "rb") as f:
                    m = pickle.load(f)
                logger.info("Loaded model from local file: %s", path)
                return m
            except Exception as e:
                logger.warning("Failed to load model file %s: %s", path, e)
    # if we reach here,
    # nothing worked
    tried = ", ".join([os.path.join(base_dir, n) for n in candidates])
    raise FileNotFoundError(
        "No model found. Tried local files: {}. "
        "And everything else.".format(tried)
    )

# load model (raises descriptive error if not available)
model = load_model_auto()

# -----------------------
# FastAPI app
# -----------------------
app = FastAPI()

# Allow cross-origin requests while testing (restrict origins in production)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],            # or ["http://localhost:5500"] for a specific origin
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -----------------------
# Serve a simple HTML page for recommendations
# -----------------------
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse

app.mount("/static", StaticFiles(directory="."), name="static")


@app.get("/", response_class=HTMLResponse)
async def root():
    with open("index.html", "r", encoding="utf-8") as f:
        return f.read()


# -----------------------
# Recommendation function
# -----------------------
# Function to get top N recommendations for a user
def get_top_n_recommendations(df, model, user_id, n=5):
    """
    Predict top-n articles for user_id using model.
    Uses Surprise trainset raw id mappings and consistent string casting.
    """
    raw_user = str(user_id)
    trainset = getattr(model, "trainset", None)

    # Popularity fallback helper
    def popular_top(k):
        popular = df['click_article_id'].astype(str).value_counts().index.tolist()
        return [int(x) if str(x).isdigit() else x for x in popular[:k]]

    if trainset is None:
        logger.warning("Model has no trainset; returning popular items")
        return popular_top(n)

    raw2inner_u = getattr(trainset, "_raw2inner_id_users", {})
    raw2inner_i = getattr(trainset, "_raw2inner_id_items", {})

    # Cold-start: unknown user -> popularity
    if raw_user not in raw2inner_u:
        logger.info("User %s unknown to model -> popularity fallback", raw_user)
        return popular_top(n)

    # Candidate items: only items known to the trained model
    candidate_items = list(raw2inner_i.keys())
    # Items user already saw (string-cast)
    user_seen = set(df[df['user_id'].astype(str) == raw_user]['click_article_id'].astype(str).unique())

    predictions = []
    for raw_item in candidate_items:
        if raw_item in user_seen:
            continue
        try:
            p = model.predict(raw_user, raw_item)
            score = float(p.est)
            predictions.append((raw_item, score))
        except Exception as e:
            # skip items that fail for any reason
            logger.debug("predict failed for user=%s item=%s: %s", raw_user, raw_item, e)
            continue

    if not predictions:
        logger.info("No predictions (all candidate items skipped) -> popularity fallback")
        return popular_top(n)

    scores = [s for _, s in predictions]
    # If model returns near-constant scores (global mean) -> popularity fallback
    if max(scores) - min(scores) < 1e-6:
        logger.info("Predictions show no variance (global mean). Using popularity fallback")
        return popular_top(n)

    # sort and return top-n (convert back to int when appropriate)
    predictions.sort(key=lambda x: x[1], reverse=True)
    top_raw = [p[0] for p in predictions[:n]]
    return [int(x) if str(x).isdigit() else x for x in top_raw]

# -----------------------
# # Post Recommendation endpoint
# # -----------------------
from pydantic import BaseModel

class RecommendRequest(BaseModel):
    user_id: int
    n: int = 5

class RecommendResponse(BaseModel):
    articles: List[int]

# Register both paths so clients with/without trailing slash won't get 404 for POST
@app.post("/recommendations", response_model=RecommendResponse)
@app.post("/recommendations/", response_model=RecommendResponse)
async def recommend_post(body: RecommendRequest):
    user_id = int(body.user_id)
    n = int(body.n)

    # validate user exists
    if not any(df['user_id'].astype(int) == user_id):   # check in full df and replaced train with df
        raise HTTPException(status_code=404, detail="User not found")

    recommended = get_top_n_recommendations(df, model, user_id, n) #replaced train with df
    if not recommended:
        raise HTTPException(status_code=404, detail="No recommendations available")
    return  {"articles": recommended}

# -----------------------
# API endpoint
# -----------------------

@app.get("/recommend/{user_id}", response_model=RecommendResponse)
async def recommend_articles(user_id: int):
    # Check user exists in training set
    if not any(df['user_id'].astype(int) == int(user_id)): #replaced train with df can also replace with trainset
        raise HTTPException(status_code=404, detail="User not found")

    recommended = get_top_n_recommendations(df, model, user_id) #replaced train with df
    if not recommended:
        raise HTTPException(status_code=404, detail="No recommendations available")
    return {"articles": recommended}

# -----------------------
# Debug endpoint
# -----------------------

@app.get("/_debug", response_class=JSONResponse)
async def debug_info(sample_user_count: int = 3, sample_item_count: int = 10):
    trainset = getattr(model, "trainset", None)
    if trainset is None:
        return {"error": "model has no trainset (maybe not a Surprise model or not trained)"}

    raw2inner_u = getattr(trainset, "_raw2inner_id_users", {})
    raw2inner_i = getattr(trainset, "_raw2inner_id_items", {})

    n_users = len(raw2inner_u)
    n_items = len(raw2inner_i)

    # dataset-level stats from the raw df used to build the dataset
    # assume 'session_size' was used as the rating in the original training
    rating_series = df['session_size'].astype(float)
    rating_stats = {
        "min": float(rating_series.min()),
        "max": float(rating_series.max()),
        "mean": float(rating_series.mean()),
        "std": float(rating_series.std()),
        "var": float(rating_series.var()),
        "n_unique": int(rating_series.nunique())
    }

    # Surprise trainset global mean (if available)
    global_mean = getattr(trainset, "global_mean", None)

    # pick sample users/items
    sample_users = list(raw2inner_u.keys())[:sample_user_count]
    sample_items = list(raw2inner_i.keys())[:sample_item_count]

    # Build small matrix of predictions and collect per-user score variance
    preds = {}
    per_user_ranges = {}
    for u in sample_users:
        preds[u] = []
        seen = set(df[df['user_id'].astype(str) == str(u)]['click_article_id'].astype(str).unique())
        scores = []
        for it in sample_items:
            if str(it) in seen:
                preds[u].append({"item": it, "seen": True})
                continue
            try:
                p = model.predict(str(u), str(it))
                score = float(p.est)
                preds[u].append({"item": it, "score": score})
                scores.append(score)
            except Exception as e:
                preds[u].append({"item": it, "error": str(e)})
        per_user_ranges[u] = {"min": min(scores) if scores else None, "max": max(scores) if scores else None,
                              "range": (max(scores)-min(scores)) if scores else None}

    info = {
        "n_users_in_trainset": n_users,
        "n_items_in_trainset": n_items,
        "trainset_global_mean": float(global_mean) if global_mean is not None else None,
        "raw_rating_stats": rating_stats,
        "sample_users": sample_users,
        "sample_items": sample_items,
        "sample_predictions": preds,
        "per_user_score_ranges": per_user_ranges
    }

    logger.info("Debug info requested: users=%d items=%d", n_users, n_items)
    return info

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
