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
# Split the data into training and test sets
# -----------------------
reader = Reader(rating_scale=(1, 130))
data = Dataset.load_from_df(df[['user_id', 'click_article_id', 'session_size']], reader)
trainset, testset = train_test_split(data, test_size=0.25, random_state=42)
# print(f"Training set size: {train.shape}\nTest set size: {test.shape}")

# ------------------------
# Load the trained model
# ------------------------
model_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "svd_algo_retrained.pkl"))
if not os.path.exists(model_path):
    raise FileNotFoundError(f"Model file not found: {model_path}")
with open(model_path, 'rb') as file:
    model = pickle.load(file)

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

def get_top_n_recommendations(df: pd.DataFrame, model, user_id: int, n: int = 5) -> List[int]:
    # Use raw ids exactly as used when training the Surprise model (often strings)
    raw_user = str(user_id)


    # Check if user exists in Surprise trainset — avoids global-mean predictions for unknown users
    try:
        known_users = getattr(model.trainset, '_raw2inner_id_users', None)
        user_known = (known_users is not None) and (raw_user in known_users)
    except Exception:
        user_known = False

    # print("user known in model:", str(raw_user) in model.trainset._raw2inner_id_users) # diagnostics
    # print("raw user:", raw_user) # diagnostics

    # check model has trainset attribute
    if not hasattr(model, 'trainset'):
        raise ValueError("Model is not trained or does not have a 'trainset' attribute")
    # print("has trainset:", hasattr(model, "trainset")) # diagnostics
    # print("number of users:", len(model.trainset._raw2inner_id_users)) # diagnostics
    # print("number of items:", len(model.trainset._raw2inner_id_items)) # diagnostics


    if not user_known:
        # Cold-start fallback: return top-n most popular articles from df
        popular = df['click_article_id'].value_counts().index.tolist()
        return [int(a) if str(a).isdigit() else a for a in popular[:n]]

    all_article_ids = df['click_article_id'].unique()
    user_article_ids = set(df[df['user_id'].astype(str) == raw_user]['click_article_id'].astype(str).unique())


    predictions = []
    for article_id in all_article_ids:
        raw_item = str(article_id)
        if raw_item in user_article_ids:
            continue
        # predict using raw ids (strings)
        try:
            pred = model.predict(raw_user, raw_item)
            score = float(pred.est)
        except Exception:
            # fallback: try other types
            try:
                pred = model.predict(int(raw_user), int(raw_item))
                score = float(pred.est)
            except Exception:
                continue
        predictions.append((article_id, score))
    
    print(model.predict(str(raw_user), str(raw_user))) # Example prediction for diagnostics

    # Diagnostics: if all scores equal, return popular fallback
    scores = [p[1] for p in predictions]
    if len(scores) and (max(scores) - min(scores) < 1e-6):
        popular = df['click_article_id'].value_counts().index.tolist()
        return [int(a) if str(a).isdigit() else a for a in popular[:n]]

    predictions.sort(key=lambda x: x[1], reverse=True)
    top_ids = [int(p[0]) if (isinstance(p[0], (int, float)) or str(p[0]).isdigit()) else p[0] for p in predictions[:n]]
    return top_ids

# -----------------------
# Post Recommendation endpoint
# -----------------------
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
