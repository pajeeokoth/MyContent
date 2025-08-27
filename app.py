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
     # Ensure consistent types
     all_article_ids = df['click_article_id'].unique()
     user_article_ids = df[df['user_id'] == user_id]['click_article_id'].unique()
     predictions = []
 
     for article_id in all_article_ids:
         if article_id not in user_article_ids:
             # Surprise model.predict expects raw ids (often strings) — cast to str for safety
             try:
                 pred = model.predict(str(user_id), str(article_id))
                 score = float(pred.est)
             except Exception:
                 # fallback: try without casting
                 pred = model.predict(user_id, article_id)
                 score = float(pred.est)
             predictions.append((article_id, score))
 
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

# Register both paths so clients with/without trailing slash won't get 404 for POST
@app.post("/recommendations", response_model=List[int])
@app.post("/recommendations/", response_model=List[int])
async def recommend_post(body: RecommendRequest):
    user_id = int(body.user_id)
    n = int(body.n)

    # validate user exists
    if not any(train['user_id'].astype(int) == user_id):
        raise HTTPException(status_code=404, detail="User not found")

    recommended = get_top_n_recommendations(train, model, user_id, n)
    if not recommended:
        raise HTTPException(status_code=404, detail="No recommendations available")
    return recommended

# -----------------------
# API endpoint
# -----------------------

@app.get("/recommend/{user_id}", response_model=List[int])
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
