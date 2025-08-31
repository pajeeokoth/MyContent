import os
import pickle
import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware # to allow cross-origin requests
from typing import List
from surprise import Dataset, Reader, SVD, accuracy
from surprise.model_selection import train_test_split, cross_validate

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
from sklearn.model_selection import train_test_split
train, test = train_test_split(df, test_size=0.2, random_state=42)
print(f"Training set size: {train.shape}\nTest set size: {test.shape}")

# ------------------------
# Load the trained model
# ------------------------
model_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "svd_model.pkl"))
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

    print("user known in model:", str(5559) in model.trainset._raw2inner_id_users) # diagnostics

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
    
    print(model.predict(str(38), str(12345))) # Example prediction for diagnostics

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
    if not any(train['user_id'].astype(int) == user_id):
        raise HTTPException(status_code=404, detail="User not found")

    recommended = get_top_n_recommendations(train, model, user_id, n)
    if not recommended:
        raise HTTPException(status_code=404, detail="No recommendations available")
    return  {"articles": recommended}

# -----------------------
# API endpoint
# -----------------------

@app.get("/recommend/{user_id}", response_model=RecommendResponse)
async def recommend_articles(user_id: int):
    # Check user exists in training set
    if not any(train['user_id'].astype(int) == int(user_id)):
        raise HTTPException(status_code=404, detail="User not found")

    recommended = get_top_n_recommendations(train, model, user_id)
    if not recommended:
        raise HTTPException(status_code=404, detail="No recommendations available")
    return recommended

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
