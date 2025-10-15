from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional
from fastapi.middleware.cors import CORSMiddleware

from classifier import BERTTaskClassifier
from recommender_wrapper import load_metadata, recommend_models
from baseline_store import get_baseline, set_baseline

app = FastAPI(title="Sustainable AI Recommender API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

CSV_PATH = "model_metadata.csv"
df = load_metadata(CSV_PATH)
categories = sorted(df["task"].dropna().unique().tolist())

classifier = BERTTaskClassifier()
classifier.fit_categories(list(categories))

class ClassifyRequest(BaseModel):
    text: str
    top_k: Optional[int] = 1

class ClassifyResponse(BaseModel):
    category: str
    score: float

class RecommendRequest(BaseModel):
    task: str
    min_acc: Optional[float] = 0.0
    max_lat: Optional[float] = None
    carbon_budget: Optional[float] = None
    priority: Optional[str] = "balanced"
    topk: Optional[int] = 3
    baseline_model: Optional[str] = None

@app.post("/classify", response_model=ClassifyResponse)
def classify(req: ClassifyRequest):
    res = classifier.predict(req.text, top_k=req.top_k)
    if not res:
        raise HTTPException(status_code=404, detail="No category matched")
    return {"category": res[0]["category"], "score": res[0]["score"]}

@app.post("/recommend")
def recommend_endpoint(req: RecommendRequest):
    task_text = req.task
    task_lower = task_text.strip().lower()
    matched = None
    for c in categories:
        if task_lower == str(c).strip().lower():
            matched = c
            break
    if matched is None:
        res = classifier.predict(task_text, top_k=1)
        matched = res[0]["category"]

    # if user did not supply a baseline model, check stored go-to model for this task
    baseline = req.baseline_model or get_baseline(matched)
    try:
        recs, ctx = recommend_models(
            task=matched,
            min_acc=req.min_acc,
            max_lat=req.max_lat,
            carbon_budget=req.carbon_budget,
            priority=req.priority,
            topk=req.topk,
            baseline_model=baseline
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

    return {"task_matched": matched, "recommendations": recs, "context": ctx}


class BaselineSetRequest(BaseModel):
    task: str
    model: str


@app.post("/baseline")
def set_baseline_endpoint(req: BaselineSetRequest):
    if not req.task or not req.model:
        raise HTTPException(status_code=400, detail="task and model are required")
    set_baseline(req.task, req.model)
    return {"status": "ok", "task": req.task, "model": req.model}


@app.get("/baseline")
def get_baseline_endpoint(task: str):
    if not task:
        raise HTTPException(status_code=400, detail="task is required")
    model = get_baseline(task)
    return {"task": task, "model": model}


@app.get("/models")
def get_models_for_task(task: str):
    if not task:
        raise HTTPException(status_code=400, detail="task is required")
    # return distinct model names for the task
    matches = df[df["task"].fillna("").str.lower() == str(task).strip().lower()]
    models = matches["model"].dropna().unique().tolist()
    return {"task": task, "models": models}

