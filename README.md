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
- `README.md` — project overview

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
