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
        "svd_algo_retrained.pkl"
    ]
    base_dir = os.path.dirname(__file__)
    for name in candidates:
        path = os.path.abspath(os.path.join(base_dir, name))
        if os.path.exists(path):
            try:
                with open(path, "rb") as f:
                    m = pickle.load(f)
                logger.info("Loaded model from local file: %s", path)
                # publish loaded path for diagnostics
                globals()["MODEL_FILE_LOADED"] = path
                return m
            except Exception as e:
                logger.warning("Failed to load model file %s: %s", path, e)

    tried = ", ".join([os.path.join(base_dir, n) for n in candidates])
    raise FileNotFoundError(
        "No model found. Tried local files: {}. "
        "And everything else.".format(tried)
    )

# load model (raises descriptive error if not available)
model = load_model_auto()

# Log quick model/trainset info on startup
try:
    logger.info("Model type: %s", type(model))
    trainset = getattr(model, "trainset", None)
    if trainset is None:
        logger.warning("Loaded model has no trainset attribute.")
    else:
        logger.info("Trainset: n_users=%d n_items=%d", trainset.n_users, trainset.n_items)
        logger.info("Trainset global_mean=%s", getattr(trainset, "global_mean", None))
    logger.info("Model file loaded: %s", globals().get("MODEL_FILE_LOADED"))
except Exception as _e:
    logger.exception("Error while logging model info: %s", _e)

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

# -----------------------
# Extended Debug endpoint
# -----------------------

@app.get("/_debug_full", response_class=JSONResponse)
async def debug_full(user_id: str = None, item_id: str = None, sample_users: int = 5, sample_items: int = 20):
    """
    Extended diagnostics:
    - Reports which model file was loaded
    - Trainset stats and global_mean
    - Rating distribution in current df (min/max/mean/std/unique)
    - Sample predictions for given user/item or for sampled users/items
    Use query params: ?user_id=123&item_id=456
    """
    info = {}
    info["model_file_loaded"] = globals().get("MODEL_FILE_LOADED")
    info["model_type"] = str(type(model))

    trainset = getattr(model, "trainset", None)
    if trainset is None:
        info["trainset"] = None
    else:
        raw2inner_u = getattr(trainset, "_raw2inner_id_users", {})
        raw2inner_i = getattr(trainset, "_raw2inner_id_items", {})
        info["trainset"] = {
            "n_users": len(raw2inner_u),
            "n_items": len(raw2inner_i),
            "global_mean": float(getattr(trainset, "global_mean", float("nan")))
        }

    # dataset rating stats if session_size exists
    if "session_size" in df.columns:
        rs = df["session_size"].astype(float)
        info["rating_stats"] = {"min": float(rs.min()), "max": float(rs.max()), "mean": float(rs.mean()), "std": float(rs.std()), "n_unique": int(rs.nunique())}
    else:
        info["rating_stats"] = None

    # check that sample ids are in trainset mappings
    raw2inner_u = getattr(trainset, "_raw2inner_id_users", {}) if trainset else {}
    raw2inner_i = getattr(trainset, "_raw2inner_id_items", {}) if trainset else {}

    # Helpers to sample
    sample_users_list = list(raw2inner_u.keys())[:sample_users] if raw2inner_u else []
    sample_items_list = list(raw2inner_i.keys())[:sample_items] if raw2inner_i else []

    info["sample_users_in_trainset"] = sample_users_list
    info["sample_items_in_trainset"] = sample_items_list

    # If explicit user_id/item_id requested, include them
    if user_id is not None:
        u = str(user_id)
        info["requested_user_in_trainset"] = u in raw2inner_u
    if item_id is not None:
        it = str(item_id)
        info["requested_item_in_trainset"] = it in raw2inner_i

    # Make a batch of predictions and compute variance
    def predict_safe(u, it):
        try:
            p = model.predict(str(u), str(it))
            return float(p.est)
        except Exception as e:
            return {"error": str(e)}

    preds = {}
    # if user_id specified, predict across sample items (or provided item_id)
    if user_id is not None:
        u = str(user_id)
        items = [str(item_id)] if item_id is not None else sample_items_list
        preds[u] = []
        for it in items:
            preds[u].append({"item": it, "pred": predict_safe(u, it)})
    else:
        # produce matrix for sample users x sample items
        for u in sample_users_list:
            preds[u] = []
            for it in sample_items_list:
                val = predict_safe(u, it)
                preds[u].append({"item": it, "pred": val})

    # compute basic stats: check if all numeric preds are identical
    flat_scores = []
    for u, lst in preds.items():
        for e in lst:
            v = e["pred"]
            if isinstance(v, float):
                flat_scores.append(v)

    info["sample_predictions"] = preds
    if flat_scores:
        info["pred_stats"] = {"min": min(flat_scores), "max": max(flat_scores), "range": max(flat_scores)-min(flat_scores)}
    else:
        info["pred_stats"] = None

    # quick sanity: show first candidate item used by app recommender
    info["first_candidate_item"] = sample_items_list[0] if sample_items_list else None

    return info

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
