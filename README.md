# Roadscapes VLM Comparison under Day and Night Conditions

This project benchmarks multiple Vision-Language Models (VLMs) on road-scene question answering using the Roadscapes dataset under different lighting conditions.

## Models Compared
- Phi-3
- Qwen2-VL
- LLaVA
- PaliGemma

## Tasks
- Object Counting
- Object Description
- Surrounding Description

## Dataset
Roadscapes dataset:
https://github.com/roadscapes/roadscapes_data/tree/main

## Objective
To compare how different VLMs perform on structured road-scene understanding tasks in **day** and **night** conditions.

## Evaluation Setup
Each model was evaluated on:
- Day images
- Night images
- Three question categories

Metrics used:
- Accuracy by task category
- Day vs Night performance
- Overall average accuracy

## Final Results
Overall model ranking based on average accuracy:
1. **Qwen2-VL** — 49.33%
2. **Phi-3** — 45.83%
3. **LLaVA** — 42.83%
4. **PaliGemma** — 40.00%

### Observations
- **Qwen2-VL** achieved the best overall performance.
- **Phi-3** and **Qwen2-VL** performed especially well on **Surrounding Description**.
- **PaliGemma** showed strong performance in **Object Counting**.
- All models showed weaker performance on **Object Description** compared to the other tasks.
- Day images generally produced higher accuracy than night images.

## Repository Structure
- `notebooks/` — experiment notebooks
- `results/` — CSV outputs and final charts
- `backend/` — FastAPI demo server and question data
- `frontend/` — browser demo for image upload and four-model comparison
- `README.md` — project overview

## Demo Web App
Run the local demo from the backend folder:

```bash
cd backend
uv run uvicorn main:app --host 127.0.0.1 --port 8000 --reload
```

Then open:

```text
http://127.0.0.1:8000
```

The demo loads questions from `backend/questions.csv`, builds the project prompt format, accepts a dropped road image, and shows side-by-side outputs for:

- `microsoft/Phi-3.5-vision-instruct`
- `Qwen/Qwen2-VL-2B-Instruct`
- `google/paligemma-3b-mix-224`
- `llava-hf/llava-1.5-7b-hf`

Current behavior is deterministic demo inference so the app is immediately usable without downloading large model weights. Replace the mock answer function in `backend/main.py` with local or hosted model calls when you are ready to run live VLM inference.

## Tools and Libraries
- Python
- Pandas
- NumPy
- Matplotlib
- Hugging Face Transformers
- Kaggle

## Author
Vishal

##Contact
vishal05.official@gmail.com
