from __future__ import annotations

import base64
import csv
import hashlib
import mimetypes
import os
from pathlib import Path
from random import Random
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel


ROOT = Path(__file__).resolve().parents[1]
FRONTEND_DIR = ROOT / "frontend"
RESULTS_DIR = ROOT / "results"
QUESTIONS_CSV = ROOT / "backend" / "questions.csv"
DAY_NIGHT_CSV = ROOT / "results" / "day_night_overall_all_4_models.csv"


MODEL_CONFIGS = [
    {
        "key": "qwen2vl",
        "label": "Qwen2-VL",
        "model_id": "Qwen/Qwen2-VL-2B-Instruct",
        "accent": "#2563eb",
    },
    {
        "key": "phi3",
        "label": "Phi-3",
        "model_id": "microsoft/Phi-3.5-vision-instruct",
        "accent": "#059669",
    },
    {
        "key": "llava",
        "label": "LLaVA",
        "model_id": "llava-hf/llava-1.5-7b-hf",
        "accent": "#d97706",
    },
    {
        "key": "paligemma",
        "label": "PaliGemma",
        "model_id": "google/paligemma-3b-mix-224",
        "accent": "#be123c",
    },
]


class AnalyzeRequest(BaseModel):
    image_data_url: str
    image_name: str | None = None
    category: str
    light: str
    question: str


def is_yes_no_question(question: str) -> bool:
    normalized = question.strip().lower()
    return normalized.startswith(("is ", "are ", "does ", "do ", "can ", "has ", "have "))


def build_prompt(category: str, light: str, question: str) -> str:
    q = str(question).strip()

    if category == "Object Counting":
        if is_yes_no_question(q):
            return f"""Look at the road image carefully.

Lighting: {light}

Rules:
- Answer only Yes or No.
- No explanation.
- No extra words.

Question: {q}"""
        return f"""Look at the road image carefully.

Lighting: {light}

Rules:
- Count the requested object carefully.
- Answer with only one integer number.
- No words.
- No explanation.
- If not visible, answer 0.

Question: {q}"""

    if category == "Object Description":
        return f"""Look at the road image carefully.

Lighting: {light}

Rules:
- Answer in 1 to 3 words only.
- No explanation.
- No sentence.

Question: {q}"""

    return f"""Look at the road image carefully.

Lighting: {light}

Rules:
- Answer in 2 to 6 words only.
- Briefly describe the surroundings.
- No explanation.
- No full sentence.

Question: {q}"""


def read_questions() -> list[dict[str, str]]:
    if not QUESTIONS_CSV.exists():
        return []

    rows: list[dict[str, str]] = []
    with QUESTIONS_CSV.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            category = row.get("category", "").strip()
            if category in {"Object Counting", "Object Description", "Surrounding Description"}:
                rows.append(
                    {
                        "filename": row.get("filename", "").strip(),
                        "category": category,
                        "question": row.get("question", "").strip(),
                        "answer": row.get("answer", "").strip(),
                    }
                )
    return rows


def read_day_night_scores() -> dict[str, dict[str, float]]:
    scores: dict[str, dict[str, float]] = {}
    if not DAY_NIGHT_CSV.exists():
        return scores

    with DAY_NIGHT_CSV.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            model = row.get("model", "").strip()
            light = row.get("light", "").strip().lower()
            accuracy = float(row.get("accuracy_percent", 0) or 0)
            scores.setdefault(model, {})[light] = accuracy
    return scores


def decode_data_url(data_url: str) -> tuple[bytes, str]:
    if "," not in data_url:
        raise HTTPException(status_code=400, detail="Image data URL is malformed.")

    header, payload = data_url.split(",", 1)
    if not header.startswith("data:image/"):
        raise HTTPException(status_code=400, detail="Please upload an image file.")

    mime_type = header.removeprefix("data:").split(";", 1)[0]
    try:
        return base64.b64decode(payload, validate=True), mime_type
    except ValueError as exc:
        raise HTTPException(status_code=400, detail="Image data could not be decoded.") from exc


def mock_answer(category: str, question: str, light: str, model_key: str, image_bytes: bytes) -> str:
    seed = hashlib.sha256(image_bytes[:4096] + f"{category}|{question}|{light}|{model_key}".encode()).hexdigest()
    rng = Random(int(seed[:16], 16))
    q = question.lower()

    if category == "Object Counting":
        if is_yes_no_question(question):
            return "Yes" if rng.random() > 0.32 else "No"
        return str(rng.randint(0, 5))

    if category == "Object Description":
        color_words = ["white", "gray", "black", "red", "blue", "silver"]
        object_words = ["car", "truck", "traffic sign", "bus", "rider", "lane marker"]
        if "color" in q:
            return rng.choice(color_words)
        if "class" in q or "object" in q:
            return rng.choice(object_words)
        return rng.choice(["road vehicle", "traffic sign", "street object"])

    if "time of day" in q:
        return light
    if "traffic density" in q:
        return rng.choice(["Low traffic", "Moderate traffic", "High traffic"])
    return rng.choice(["urban road scene", "clear roadway", "busy street", "open road"])


def model_score(label: str, light: str, scores: dict[str, dict[str, float]]) -> float:
    return scores.get(label, {}).get(light.lower(), 0.0)


app = FastAPI(title="Roadscapes VLM Demo")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/config")
def config() -> dict[str, Any]:
    questions = read_questions()
    categories = sorted({row["category"] for row in questions})
    return {
        "models": MODEL_CONFIGS,
        "categories": categories,
        "question_count": len(questions),
        "mode": "demo",
        "note": "Demo mode returns deterministic sample outputs. Wire run_model() to local or hosted VLM inference for live model calls.",
    }


@app.get("/api/questions")
def questions(category: str | None = None, limit: int = 80) -> dict[str, Any]:
    rows = read_questions()
    if category:
        rows = [row for row in rows if row["category"] == category]

    unique: list[dict[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for row in rows:
        key = (row["category"], row["question"])
        if key not in seen:
            seen.add(key)
            unique.append(row)
        if len(unique) >= limit:
            break

    return {"questions": unique}


@app.post("/api/analyze")
def analyze(payload: AnalyzeRequest) -> dict[str, Any]:
    image_bytes, mime_type = decode_data_url(payload.image_data_url)
    if len(image_bytes) > 12 * 1024 * 1024:
        raise HTTPException(status_code=413, detail="Image is larger than 12 MB.")

    category = payload.category.strip()
    light = payload.light.strip().lower()
    question = payload.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="Choose or type a question first.")
    if light not in {"day", "night"}:
        raise HTTPException(status_code=400, detail="Lighting must be day or night.")

    scores = read_day_night_scores()
    prompt = build_prompt(category, light, question)
    results = []
    for model in MODEL_CONFIGS:
        answer = mock_answer(category, question, light, model["key"], image_bytes)
        accuracy = model_score(model["label"], light, scores)
        confidence = min(98, max(42, round(accuracy + Random(model["key"] + question).randint(20, 36), 1)))
        results.append(
            {
                **model,
                "answer": answer,
                "confidence": confidence,
                "benchmark_accuracy": accuracy,
                "latency_ms": Random(model["key"] + str(len(image_bytes))).randint(820, 2400),
            }
        )

    return {
        "image": {
            "name": payload.image_name or "uploaded image",
            "bytes": len(image_bytes),
            "mime_type": mime_type,
            "extension": mimetypes.guess_extension(mime_type) or "",
        },
        "category": category,
        "light": light,
        "question": question,
        "prompt": prompt,
        "results": results,
    }


@app.get("/")
def index() -> FileResponse:
    return FileResponse(FRONTEND_DIR / "index.html")


if RESULTS_DIR.exists():
    app.mount("/results", StaticFiles(directory=RESULTS_DIR), name="results")

app.mount("/", StaticFiles(directory=FRONTEND_DIR), name="frontend")
