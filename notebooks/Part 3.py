# ==================== CELL 1 ====================
# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session

# ==================== CELL 2 ====================
!pip install -q --upgrade transformers accelerate sentencepiece
!pip install -q sentence-transformers huggingface_hub

# ==================== CELL 3 ====================
import os
import re
import gc
import random
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path
from PIL import Image
from tqdm.auto import tqdm

import torch
from transformers import (
    AutoProcessor,
    Blip2Processor,
    Blip2ForConditionalGeneration,
    PaliGemmaForConditionalGeneration
)

warnings.filterwarnings("ignore")

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float16 if DEVICE == "cuda" else torch.float32

OUTPUT_DIR = "/kaggle/working/vlm_day_night_results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

CSV_PATH = "/kaggle/working/vqa_dataset_test_mapped.csv"

# start small if needed
SAMPLE_PER_CATEGORY_PER_LIGHT = 100

BLIP2_MODEL_ID = "Salesforce/blip2-opt-2.7b"
PALIGEMMA_MODEL_ID = "google/paligemma-3b-mix-224"

print("DEVICE:", DEVICE)
if DEVICE == "cuda":
    print("GPU:", torch.cuda.get_device_name(0))

# ==================== CELL 4 ====================
if not os.path.exists("/kaggle/working/roadscapes_data"):
    !git clone https://github.com/roadscapes/roadscapes_data.git /kaggle/working/roadscapes_data

csv_path = "/kaggle/working/roadscapes_data/vqa_dataset_test.csv"
image_root = Path("/kaggle/working/roadscapes_data/image_data/images")

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

# ==================== CELL 5 ====================
df = pd.read_csv(CSV_PATH)

required_cols = ["filename", "question", "answer", "category", "full_image_path"]
missing = [c for c in required_cols if c not in df.columns]
if missing:
    raise ValueError(f"Missing columns: {missing}")

df = df[df["full_image_path"].notna()].copy()
df = df[df["full_image_path"].apply(lambda x: os.path.exists(str(x)))].copy()

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

df["category_std"] = df["category"].apply(normalize_category)
df["light"] = df["filename"].apply(get_light)

target_categories = [
    "Object Counting",
    "Object Description",
    "Surrounding Description"
]

df = df[df["category_std"].isin(target_categories)].copy()
df = df[df["light"].isin(["day", "night"])].copy()

print("Rows:", len(df))
print(df["light"].value_counts())
print(df["category_std"].value_counts())

# ==================== CELL 6 ====================
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

# ==================== CELL 7 ====================
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

Question: {q}"""
        return f"""Look at the road image carefully.

Lighting: {light}

Rules:
- Answer with only one integer.
- No explanation.
- If not visible, answer 0.

Question: {q}"""

    if category == "Object Description":
        return f"""Look at the road image carefully.

Lighting: {light}

Rules:
- Answer in 1 to 3 words.
- No explanation.

Question: {q}"""

    return f"""Look at the road image carefully.

Lighting: {light}

Rules:
- Answer in 2 to 6 words.
- No explanation.

Question: {q}"""

def clean_prediction(text: str, question: str = "", category: str = "") -> str:
    text = str(text).strip().lower()
    text = re.sub(r"\s+", " ", text)

    for bad in ["there are", "there is", "it is", ".", ",", ":", ";"]:
        text = text.replace(bad, "")

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

def normalized_correct(pred: str, gt: str, category: str, question: str = "") -> bool:
    pred = clean_prediction(pred, question, category)
    gt = clean_prediction(gt, question, category)

    if pred.startswith("error:"):
        return False

    if category == "Object Counting":
        return pred == gt

    return pred == gt or gt in pred or pred in gt

def validate_question(q: str) -> bool:
    return isinstance(q, str) and len(q.strip()) > 0

# ==================== CELL 8 ====================
def clear_memory():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

# ==================== CELL 9 ====================
def load_blip2():
    processor = Blip2Processor.from_pretrained(BLIP2_MODEL_ID)

    model = Blip2ForConditionalGeneration.from_pretrained(
        BLIP2_MODEL_ID,
        torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32,
        low_cpu_mem_usage=True
    )

    model = model.to(DEVICE)
    model.eval()
    return processor, model

def query_blip2(processor, model, image_path, prompt, category="Object Counting"):
    try:
        image = Image.open(image_path).convert("RGB")
        image = image.resize((224, 224))

        short_prompt = prompt.split("Question:")[-1].strip()

        inputs = processor(
            images=image,
            text=short_prompt,
            return_tensors="pt"
        )

        inputs = {k: v.to(DEVICE) if hasattr(v, "to") else v for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=8,
                do_sample=False
            )

        pred = processor.batch_decode(outputs, skip_special_tokens=True)[0].strip().lower()

        if short_prompt.lower() in pred:
            pred = pred.replace(short_prompt.lower(), "").strip()

        return clean_prediction(pred, short_prompt, category)

    except Exception as e:
        return f"ERROR: {e}"

# ==================== CELL 10 ====================
clear_memory()
blip2_processor, blip2_model = load_blip2()
print("BLIP-2 loaded")

# ==================== CELL 11 ====================
test_row = sampled_splits[("Object Counting", "day")].iloc[0]
test_prompt = build_prompt("Object Counting", "day", test_row["question"])

pred = query_blip2(
    blip2_processor,
    blip2_model,
    test_row["full_image_path"],
    test_prompt,
    "Object Counting"
)

print("Question:", test_row["question"])
print("Ground truth:", test_row["answer"])
print("Prediction:", pred)
print("Correct:", normalized_correct(pred, test_row["answer"], "Object Counting", test_row["question"]))

# ==================== CELL 12 ====================
blip2_results = []

for category in target_categories:
    for light in ["day", "night"]:
        part = sampled_splits[(category, light)].copy()

        print(f"\nRunning BLIP-2 | {category} | {light} | samples = {len(part)}")

        for _, row in tqdm(part.iterrows(), total=len(part), desc=f"{category}-{light}"):
            question = row["question"]
            gt = row["answer"]
            image_path = row["full_image_path"]

            if not validate_question(question):
                continue

            prompt = build_prompt(category, light, question)

            pred = query_blip2(
                blip2_processor,
                blip2_model,
                image_path,
                prompt,
                category
            )

            blip2_results.append({
                "model": "blip2",
                "category": category,
                "light": light,
                "filename": row["filename"],
                "question": question,
                "ground_truth": gt,
                "prediction": pred,
                "is_correct": normalized_correct(pred, gt, category, question)
            })

blip2_results_df = pd.DataFrame(blip2_results)
print("Total evaluated:", len(blip2_results_df))
print(blip2_results_df.head())

# ==================== CELL 13 ====================
blip2_summary = (
    blip2_results_df
    .groupby(["category", "light"])["is_correct"]
    .mean()
    .reset_index()
)

blip2_summary["accuracy_percent"] = (blip2_summary["is_correct"] * 100).round(2)

print(blip2_summary)

os.makedirs(OUTPUT_DIR, exist_ok=True)

blip2_results_df.to_csv(os.path.join(OUTPUT_DIR, "blip2_results.csv"), index=False)
blip2_summary.to_csv(os.path.join(OUTPUT_DIR, "blip2_accuracy.csv"), index=False)

print("\nBLIP-2 files saved")
print(os.listdir(OUTPUT_DIR))

# ==================== CELL 14 ====================
for category in target_categories:
    sub = blip2_summary[blip2_summary["category"] == category].copy()

    lights = sub["light"].tolist()
    scores = sub["accuracy_percent"].tolist()

    plt.figure(figsize=(6, 4))
    plt.bar(lights, scores)
    plt.title(f"BLIP-2 | {category} | Day vs Night")
    plt.ylabel("Accuracy (%)")
    plt.ylim(0, 100)
    plt.tight_layout()

    out_png = os.path.join(
        OUTPUT_DIR,
        f"blip2_{category.replace(' ', '_').lower()}_day_night.png"
    )
    plt.savefig(out_png, dpi=200)
    plt.show()

print("BLIP-2 charts saved")

# ==================== CELL 15 ====================
!pip install -q --upgrade transformers==4.47.1 accelerate sentencepiece

# ==================== CELL 16 ====================
import torch
import gc
from PIL import Image
from transformers import LlavaForConditionalGeneration, AutoProcessor

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
LLAVA_MODEL_ID = "llava-hf/llava-1.5-7b-hf"

def clear_memory():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

# ==================== CELL 17 ====================
def load_llava():
    processor = AutoProcessor.from_pretrained(LLAVA_MODEL_ID)

    model = LlavaForConditionalGeneration.from_pretrained(
        LLAVA_MODEL_ID,
        torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32,
        low_cpu_mem_usage=True
    ).to(DEVICE)

    model.eval()
    return processor, model

# ==================== CELL 18 ====================
def query_llava(processor, model, image_path, prompt):
    try:
        image = Image.open(image_path).convert("RGB")
        image = image.resize((336, 336))

        full_prompt = f"USER: <image>\n{prompt}\nASSISTANT:"

        inputs = processor(
            text=full_prompt,
            images=image,
            return_tensors="pt"
        )

        inputs = {
            k: v.to(DEVICE) if hasattr(v, "to") else v
            for k, v in inputs.items()
        }

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=16,
                do_sample=False
            )

        pred = processor.batch_decode(outputs, skip_special_tokens=True)[0]

        if "ASSISTANT:" in pred:
            pred = pred.split("ASSISTANT:")[-1]

        return pred.strip().lower()

    except Exception as e:
        return f"ERROR: {e}"

# ==================== CELL 19 ====================
clear_memory()
llava_processor, llava_model = load_llava()
print("LLaVA loaded")

# ==================== CELL 20 ====================
import os
import pandas as pd
from pathlib import Path

CSV_PATH = "/kaggle/working/vqa_dataset_test_mapped.csv"

if not os.path.exists(CSV_PATH):
    if not os.path.exists("/kaggle/working/roadscapes_data"):
        !git clone https://github.com/roadscapes/roadscapes_data.git /kaggle/working/roadscapes_data

    csv_path = "/kaggle/working/roadscapes_data/vqa_dataset_test.csv"
    image_root = Path("/kaggle/working/roadscapes_data/image_data/images")

    image_index = {}
    for ext in ["*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG"]:
        for path in image_root.rglob(ext):
            image_index[path.name] = str(path)

    df_map = pd.read_csv(csv_path)

    possible_cols = ["filename", "image", "image_name", "img", "image_path", "Image"]
    image_col = None
    for col in possible_cols:
        if col in df_map.columns:
            image_col = col
            break

    if image_col is None:
        raise ValueError(f"No image column found. Available columns: {df_map.columns.tolist()}")

    df_map["full_image_path"] = df_map[image_col].astype(str).apply(
        lambda x: image_index.get(os.path.basename(x.strip()), None)
    )

    df_map.to_csv(CSV_PATH, index=False)
    print("Saved:", CSV_PATH)
else:
    print("Mapped CSV already exists:", CSV_PATH)

# ==================== CELL 21 ====================
import os
import pandas as pd

SAMPLE_PER_CATEGORY_PER_LIGHT = 100

df = pd.read_csv(CSV_PATH)

df = df[df["full_image_path"].notna()].copy()
df = df[df["full_image_path"].apply(lambda x: os.path.exists(str(x)))].copy()

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

df["category_std"] = df["category"].apply(normalize_category)
df["light"] = df["filename"].apply(get_light)

target_categories = [
    "Object Counting",
    "Object Description",
    "Surrounding Description"
]

sampled_splits = {}

for cat in target_categories:
    for light in ["day", "night"]:
        subset = df[(df["category_std"] == cat) & (df["light"] == light)]

        if len(subset) == 0:
            continue

        sample_n = min(SAMPLE_PER_CATEGORY_PER_LIGHT, len(subset))
        sampled = subset.sample(n=sample_n, random_state=42)

        sampled_splits[(cat, light)] = sampled.reset_index(drop=True)

        print(f"{cat} | {light}: {len(sampled)} samples")

# ==================== CELL 22 ====================
import re

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
        else:
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
        else:
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

# ==================== CELL 23 ====================
test_row = sampled_splits[("Object Counting", "day")].iloc[0]

prompt = build_prompt("Object Counting", "day", test_row["question"])

pred = query_llava(
    llava_processor,
    llava_model,
    test_row["full_image_path"],
    prompt
)

print("Question:", test_row["question"])
print("Ground truth:", test_row["answer"])
print("Prediction:", pred)
print("Correct:", normalized_correct(pred, test_row["answer"], "Object Counting", test_row["question"]))

# ==================== CELL 24 ====================
for i in range(5):
    test_row = sampled_splits[("Object Counting", "day")].iloc[i]

    prompt = build_prompt("Object Counting", "day", test_row["question"])

    pred = query_llava(
        llava_processor,
        llava_model,
        test_row["full_image_path"],
        prompt
    )

    print(f"\nSample {i+1}")
    print("Question:", test_row["question"])
    print("Ground truth:", test_row["answer"])
    print("Prediction:", pred)
    print("Correct:", normalized_correct(pred, test_row["answer"], "Object Counting", test_row["question"]))

# ==================== CELL 25 ====================
llava_results = []

for category in target_categories:
    for light in ["day", "night"]:
        part = sampled_splits[(category, light)].copy()

        print(f"\nRunning LLaVA | {category} | {light} | samples = {len(part)}")

        for _, row in part.iterrows():
            question = row["question"]
            gt = row["answer"]
            image_path = row["full_image_path"]

            if not validate_question(question):
                continue

            prompt = build_prompt(category, light, question)

            pred = query_llava(
                llava_processor,
                llava_model,
                image_path,
                prompt
            )

            llava_results.append({
                "model": "llava",
                "category": category,
                "light": light,
                "filename": row["filename"],
                "question": question,
                "ground_truth": gt,
                "prediction": pred,
                "is_correct": normalized_correct(pred, gt, category, question)
            })

llava_results_df = pd.DataFrame(llava_results)

print("Total evaluated:", len(llava_results_df))
print(llava_results_df.head())

# ==================== CELL 26 ====================
import os

OUTPUT_DIR = "/kaggle/working/vlm_day_night_results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("Using OUTPUT_DIR:", OUTPUT_DIR)

# ==================== CELL 27 ====================
llava_summary = (
    llava_results_df
    .groupby(["category", "light"])["is_correct"]
    .mean()
    .reset_index()
)

llava_summary["accuracy_percent"] = (llava_summary["is_correct"] * 100).round(2)

print(llava_summary)

os.makedirs(OUTPUT_DIR, exist_ok=True)

llava_results_df.to_csv(os.path.join(OUTPUT_DIR, "llava_results.csv"), index=False)
llava_summary.to_csv(os.path.join(OUTPUT_DIR, "llava_accuracy.csv"), index=False)

print("Saved files:")
print(os.listdir(OUTPUT_DIR))

# ==================== CELL 28 ====================
import matplotlib.pyplot as plt

# ==================== CELL 29 ====================
import matplotlib.pyplot as plt
import os

for category in target_categories:
    sub = llava_summary[llava_summary["category"] == category].copy()

    lights = sub["light"].tolist()
    scores = sub["accuracy_percent"].tolist()

    plt.figure(figsize=(6, 4))
    plt.bar(lights, scores)
    plt.title(f"LLaVA | {category} | Day vs Night")
    plt.ylabel("Accuracy (%)")
    plt.ylim(0, 100)
    plt.tight_layout()

    out_png = os.path.join(
        OUTPUT_DIR,
        f"llava_{category.replace(' ', '_').lower()}_day_night.png"
    )
    plt.savefig(out_png, dpi=200)
    plt.show()

print("LLaVA charts saved")
print(os.listdir(OUTPUT_DIR))

# ==================== CELL 30 ====================
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

OUTPUT_DIR = "/kaggle/working/vlm_day_night_results"

# load saved summaries
blip2_summary = pd.read_csv(os.path.join(OUTPUT_DIR, "blip2_accuracy.csv"))
llava_summary = pd.read_csv(os.path.join(OUTPUT_DIR, "llava_accuracy.csv"))

# add model names
blip2_summary["model"] = "BLIP-2"
llava_summary["model"] = "LLaVA"

# combine
compare_df = pd.concat([blip2_summary, llava_summary], ignore_index=True)

# keep clean columns
compare_df = compare_df[["model", "category", "light", "accuracy_percent"]]

# save combined table
compare_csv = os.path.join(OUTPUT_DIR, "blip2_vs_llava_comparison.csv")
compare_df.to_csv(compare_csv, index=False)

print("COMPARISON TABLE")
print(compare_df)
print("\nSaved:", compare_csv)

# ==================== CELL 31 ====================
import os
import matplotlib.pyplot as plt
import numpy as np

for category in compare_df["category"].unique():
    sub = compare_df[compare_df["category"] == category].copy()

    pivot = sub.pivot(index="model", columns="light", values="accuracy_percent").reset_index()

    x = np.arange(len(pivot))
    width = 0.35

    plt.figure(figsize=(7, 4))
    plt.bar(x - width/2, pivot["day"], width, label="Day")
    plt.bar(x + width/2, pivot["night"], width, label="Night")

    plt.xticks(x, pivot["model"])
    plt.ylabel("Accuracy (%)")
    plt.title(f"BLIP-2 vs LLaVA | {category}")
    plt.ylim(0, 100)
    plt.legend()
    plt.tight_layout()

    out_png = os.path.join(
        OUTPUT_DIR,
        f"blip2_vs_llava_{category.replace(' ', '_').lower()}.png"
    )
    plt.savefig(out_png, dpi=200)
    plt.show()

    print("Saved:", out_png)

# ==================== CELL 32 ====================
import os
import matplotlib.pyplot as plt
import numpy as np

fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=True)

categories = compare_df["category"].unique()

for ax, category in zip(axes, categories):
    sub = compare_df[compare_df["category"] == category].copy()
    pivot = sub.pivot(index="model", columns="light", values="accuracy_percent").reset_index()

    x = np.arange(len(pivot))
    width = 0.35

    ax.bar(x - width/2, pivot["day"], width, label="Day")
    ax.bar(x + width/2, pivot["night"], width, label="Night")
    ax.set_xticks(x)
    ax.set_xticklabels(pivot["model"])
    ax.set_title(category)
    ax.set_ylabel("Accuracy (%)")
    ax.set_ylim(0, 100)

axes[0].legend()
plt.tight_layout()

combined_png = os.path.join(OUTPUT_DIR, "blip2_vs_llava_all_categories.png")
plt.savefig(combined_png, dpi=200)
plt.show()

print("Saved:", combined_png)

# ==================== CELL 33 ====================
import os

OUTPUT_DIR = "/kaggle/working/vlm_day_night_results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# save comparison table
compare_csv = os.path.join(OUTPUT_DIR, "blip2_vs_llava_comparison.csv")
compare_df.to_csv(compare_csv, index=False)

print("Comparison saved at:")
print(compare_csv)

print("\nFiles in folder:")
print(os.listdir(OUTPUT_DIR))

# ==================== CELL 34 ====================
combined_png = os.path.join(OUTPUT_DIR, "blip2_vs_llava_all_categories.png")
plt.savefig(combined_png, dpi=200)
print("Chart saved at:")
print(combined_png)

# ==================== CELL 35 ====================
print(os.path.exists(os.path.join(OUTPUT_DIR, "blip2_vs_llava_comparison.csv")))
print(os.path.exists(os.path.join(OUTPUT_DIR, "blip2_vs_llava_all_categories.png")))

# ==================== CELL 36 ====================
import os
import zipfile

source_dir = "/kaggle/working/vlm_day_night_results"
zip_path = "/kaggle/working/vlm_day_night_results_backup.zip"

with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zipf:
    for root, _, files in os.walk(source_dir):
        for file in files:
            full_path = os.path.join(root, file)
            arcname = os.path.relpath(full_path, source_dir)
            zipf.write(full_path, arcname)

print("Backup zip created:", zip_path)
print("Exists:", os.path.exists(zip_path))

# ==================== CELL 37 ====================
import os
print(os.listdir("/kaggle/working/vlm_day_night_results"))

# ==================== CELL 38 ====================
import os
print(os.path.exists("/kaggle/working/vlm_day_night_results_backup.zip"))
print(os.listdir("/kaggle/working"))

