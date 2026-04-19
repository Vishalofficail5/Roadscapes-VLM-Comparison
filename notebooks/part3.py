import os
import re
import gc
import random
import shutil
import warnings
import subprocess
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path
from PIL import Image
from tqdm.auto import tqdm

import torch
from transformers import AutoProcessor, PaliGemmaForConditionalGeneration
from huggingface_hub import login, whoami, hf_hub_download

warnings.filterwarnings("ignore")

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float16 if DEVICE == "cuda" else torch.float32

CSV_PATH = "/kaggle/working/vqa_dataset_test_mapped.csv"
OUTPUT_DIR = "/kaggle/working/vlm_day_night_results"
ROADSCAPES_DIR = "/kaggle/working/roadscapes_data"
SAMPLE_PER_CATEGORY_PER_LIGHT = 100
PALIGEMMA_MODEL_ID = "google/paligemma-3b-mix-224"

os.makedirs(OUTPUT_DIR, exist_ok=True)

print("DEVICE:", DEVICE)
if DEVICE == "cuda":
    print("GPU:", torch.cuda.get_device_name(0))


def install_dependencies():
    """Optional helper for Kaggle/Colab."""
    packages = [
        "transformers==4.45.2",
        "accelerate",
        "sentencepiece",
        "huggingface_hub",
    ]
    subprocess.run(
        ["python", "-m", "pip", "install", "-q", "--upgrade", *packages],
        check=True,
    )


def setup_huggingface(hf_token: str):
    login(token=hf_token)
    print("Logged in to Hugging Face")
    print(whoami())

    path = hf_hub_download(
        repo_id=PALIGEMMA_MODEL_ID,
        filename="config.json",
        token=hf_token,
    )
    print("PaliGemma access approved")
    print(path)


def clone_roadscapes_repo():
    if not os.path.exists(ROADSCAPES_DIR):
        subprocess.run(
            ["git", "clone", "https://github.com/roadscapes/roadscapes_data.git", ROADSCAPES_DIR],
            check=True,
        )


def prepare_dataset_mapping():
    clone_roadscapes_repo()

    csv_path = os.path.join(ROADSCAPES_DIR, "vqa_dataset_test.csv")
    image_root = Path(os.path.join(ROADSCAPES_DIR, "image_data/images"))

    image_index = {}
    for ext in ["*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG"]:
        for path in image_root.rglob(ext):
            image_index[path.name] = str(path)

    print("Indexed images:", len(image_index))

    df_map = pd.read_csv(csv_path)
    print("CSV rows:", len(df_map))
    print("Columns:", df_map.columns.tolist())

    possible_cols = ["filename", "image", "image_name", "img", "image_path", "Image"]
    image_col = None

    for col in possible_cols:
        if col in df_map.columns:
            image_col = col
            break

    if image_col is None:
        raise ValueError(f"No image column found. Available columns: {df_map.columns.tolist()}")

    print("Using image column:", image_col)

    df_map["full_image_path"] = df_map[image_col].astype(str).apply(
        lambda x: image_index.get(os.path.basename(x.strip()), None)
    )

    print("Matched rows:", df_map["full_image_path"].notna().sum())

    df_map.to_csv(CSV_PATH, index=False)
    print("Saved:", CSV_PATH)


def normalize_category(cat):
    c = str(cat).strip().lower().replace("_", " ")
    if "count" in c:
        return "Object Counting"
    if "surround" in c:
        return "Surrounding Description"
    if "description" in c:
        return "Object Description"
    return str(cat)


def get_light(filename):
    return "night" if "night" in str(filename).lower() else "day"


def load_and_filter_dataset():
    df = pd.read_csv(CSV_PATH)

    required_cols = ["filename", "question", "answer", "category", "full_image_path"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    df = df[df["full_image_path"].notna()].copy()
    df = df[df["full_image_path"].apply(lambda x: os.path.exists(str(x)))].copy()

    df["category_std"] = df["category"].apply(normalize_category)
    df["light"] = df["filename"].apply(get_light)

    target_categories = [
        "Object Counting",
        "Object Description",
        "Surrounding Description",
    ]

    df = df[df["category_std"].isin(target_categories)].copy()
    df = df[df["light"].isin(["day", "night"])].copy()

    print("Rows:", len(df))
    print(df["light"].value_counts())
    print(df["category_std"].value_counts())
    return df, target_categories


def create_sample_splits(df, target_categories):
    sampled_splits = {}

    for cat in target_categories:
        for light in ["day", "night"]:
            subset = df[(df["category_std"] == cat) & (df["light"] == light)].copy()
            if len(subset) == 0:
                continue

            sample_n = min(SAMPLE_PER_CATEGORY_PER_LIGHT, len(subset))
            sampled = subset.sample(n=sample_n, random_state=SEED)
            sampled_splits[(cat, light)] = sampled.reset_index(drop=True)
            print(f"{cat} | {light}: {len(sampled)} samples")

    return sampled_splits


def is_yes_no_question(question: str) -> bool:
    q = str(question).strip().lower()
    return q.startswith("is ") or q.startswith("are ") or " more " in q


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


def clean_prediction(text: str, question: str = "", category: str = "") -> str:
    text = str(text).strip().lower()
    text = re.sub(r"\s+", " ", text)

    for bad in ["there are", "there is", "it is", ".", ",", ":", ";"]:
        text = text.replace(bad, "")

    text = text.strip()

    if is_yes_no_question(question):
        if "yes" in text:
            return "yes"
        if "no" in text:
            return "no"

    if category == "Object Counting" and not is_yes_no_question(question):
        nums = re.findall(r"\d+", text)
        if nums:
            return nums[0]

    return text.strip()


def extract_first_number(text: str):
    m = re.search(r"\d+", str(text))
    return m.group(0) if m else None


def normalized_correct(pred: str, gt: str, category: str, question: str = "") -> bool:
    pred = clean_prediction(pred, question, category)
    gt = clean_prediction(gt, question, category)

    if pred.startswith("error:"):
        return False

    if category == "Object Counting":
        if is_yes_no_question(question):
            return pred == gt
        p = extract_first_number(pred)
        g = extract_first_number(gt)
        return p is not None and g is not None and p == g

    if pred == gt:
        return True
    if gt in pred or pred in gt:
        return True

    pred_words = set(pred.split())
    gt_words = set(gt.split())
    return len(pred_words & gt_words) >= 1


def validate_question(q: str) -> bool:
    return isinstance(q, str) and len(q.strip()) > 0


def clear_memory():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def load_paligemma():
    processor = AutoProcessor.from_pretrained(PALIGEMMA_MODEL_ID)
    model = PaliGemmaForConditionalGeneration.from_pretrained(
        PALIGEMMA_MODEL_ID,
        torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32,
        low_cpu_mem_usage=True,
    ).to(DEVICE)
    model.eval()
    return processor, model


def query_paligemma(processor, model, image_path, prompt, category="Object Counting"):
    try:
        image = Image.open(image_path).convert("RGB")
        image = image.resize((224, 224))

        short_prompt = prompt.split("Question:")[-1].strip()

        inputs = processor(text=short_prompt, images=image, return_tensors="pt")
        inputs = {k: v.to(DEVICE) if hasattr(v, "to") else v for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=12, do_sample=False)

        pred = processor.batch_decode(outputs, skip_special_tokens=True)[0].strip().lower()
        return clean_prediction(pred, short_prompt, category)
    except Exception as e:
        return f"ERROR: {e}"


def run_single_test(processor, model, sampled_splits):
    test_row = sampled_splits[("Object Counting", "day")].iloc[0]
    test_prompt = build_prompt("Object Counting", "day", test_row["question"])

    pred = query_paligemma(
        processor,
        model,
        test_row["full_image_path"],
        test_prompt,
        "Object Counting",
    )

    print("Question:", test_row["question"])
    print("Ground truth:", test_row["answer"])
    print("Prediction:", pred)
    print(
        "Correct:",
        normalized_correct(pred, test_row["answer"], "Object Counting", test_row["question"]),
    )


def evaluate_model(processor, model, sampled_splits, target_categories):
    paligemma_results = []

    for category in target_categories:
        for light in ["day", "night"]:
            part = sampled_splits.get((category, light))
            if part is None or part.empty:
                continue

            print(f"\nRunning PaliGemma | {category} | {light} | samples = {len(part)}")

            for _, row in tqdm(part.iterrows(), total=len(part), desc=f"{category}-{light}"):
                question = row["question"]
                gt = row["answer"]
                image_path = row["full_image_path"]

                if not validate_question(question):
                    continue

                prompt = build_prompt(category, light, question)
                pred = query_paligemma(processor, model, image_path, prompt, category)

                paligemma_results.append(
                    {
                        "model": "paligemma",
                        "category": category,
                        "light": light,
                        "filename": row["filename"],
                        "question": question,
                        "ground_truth": gt,
                        "prediction": pred,
                        "is_correct": normalized_correct(pred, gt, category, question),
                    }
                )

    paligemma_results_df = pd.DataFrame(paligemma_results)
    print("Total evaluated:", len(paligemma_results_df))
    print(paligemma_results_df.head())
    return paligemma_results_df


def save_summary_and_results(results_df):
    paligemma_summary = (
        results_df.groupby(["category", "light"])["is_correct"].mean().reset_index()
    )
    paligemma_summary["accuracy_percent"] = (paligemma_summary["is_correct"] * 100).round(2)

    print(paligemma_summary)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    results_df.to_csv(os.path.join(OUTPUT_DIR, "paligemma_results.csv"), index=False)
    paligemma_summary.to_csv(os.path.join(OUTPUT_DIR, "paligemma_accuracy.csv"), index=False)

    print("Saved files:")
    print(os.listdir(OUTPUT_DIR))
    return paligemma_summary


def save_charts(summary_df, target_categories):
    for category in target_categories:
        sub = summary_df[summary_df["category"] == category].copy()
        if sub.empty:
            continue

        lights = sub["light"].tolist()
        scores = sub["accuracy_percent"].tolist()

        plt.figure(figsize=(6, 4))
        plt.bar(lights, scores)
        plt.title(f"PaliGemma | {category} | Day vs Night")
        plt.ylabel("Accuracy (%)")
        plt.ylim(0, 100)
        plt.tight_layout()

        out_png = os.path.join(
            OUTPUT_DIR,
            f"paligemma_{category.replace(' ', '_').lower()}_day_night.png",
        )
        plt.savefig(out_png, dpi=200)
        plt.close()

    print("PaliGemma charts saved")


def main(hf_token: str):
    setup_huggingface(hf_token)
    prepare_dataset_mapping()
    df, target_categories = load_and_filter_dataset()
    sampled_splits = create_sample_splits(df, target_categories)

    clear_memory()
    processor, model = load_paligemma()
    print("PaliGemma loaded")

    run_single_test(processor, model, sampled_splits)
    results_df = evaluate_model(processor, model, sampled_splits, target_categories)
    summary_df = save_summary_and_results(results_df)
    save_charts(summary_df, target_categories)

    print("Files in result folder:")
    print(os.listdir(OUTPUT_DIR))


if __name__ == "__main__":
    # Replace this with your own Hugging Face token before running.
    HF_TOKEN = os.environ.get("HF_TOKEN", "")
    if not HF_TOKEN:
        raise ValueError("Set your Hugging Face token in the HF_TOKEN environment variable.")

    main(HF_TOKEN)
