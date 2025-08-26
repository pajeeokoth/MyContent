import os
import pickle
import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from fastapi.templating import Jinja2Templates
from typing import List
from surprise import Dataset, Reader, SVD, accuracy
from surprise.model_selection import train_test_split, cross_validate

# Load and combine CSVs
directory = '../data/clicks_ext/clicks/'
df = pd.concat(
    [pd.read_csv(os.path.join(directory, file)) for file in os.listdir(directory) if file.endswith('.csv')],
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
with open('svd_model.pkl', 'rb') as file:
    model = pickle.load(file)

# -----------------------
# FastAPI app
# -----------------------
app = FastAPI()


#####################################################################
# -----------------------
# Serve a simple HTML page for recommendations
# -----------------------
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse

app.mount("/static", StaticFiles(directory="."), name="static")


@app.get("/", response_class=HTMLResponse)
async def root():
    with open("index.html") as f:
        return f.read()

#####################################################################

# -----------------------
# Recommendation function
# -----------------------

def get_top_n_recommendations(df, model, user_id, n=5):
    all_article_ids = df['click_article_id'].unique()
    user_article_ids = df[df['user_id'] == user_id]['click_article_id'].unique()
    predictions = []

    for article_id in all_article_ids:
        if article_id not in user_article_ids:
            pred = model.predict(user_id, article_id)
            predictions.append((article_id, pred.est))

    predictions.sort(key=lambda x: x[1], reverse=True)
    return [pred[0] for pred in predictions[:n]]

# -----------------------
# API endpoint
# -----------------------
# @app.get("/recommend/{user_id}", response_model=List[int])
def recommend_articles(user_id: int):
    if user_id not in train['user_id'].unique():
        raise HTTPException(status_code=404, detail="User not found")

    recommended = get_top_n_recommendations(train, model, user_id)
    return recommended

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
