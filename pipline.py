#!/usr/bin/env python3
import argparse
import os
from pathlib import Path
import ast

import numpy as np
import pandas as pd
import ollama


# ----------------------------
# Task 1: prompt image -> text
# ----------------------------
def prompt_image(image_path: str, model: str = "ministral-3:3b") -> str:
    prompt = "Describe briefly what happens in this image, in English."
    r = ollama.generate(model=model, prompt=prompt, images=[image_path])
    # ollama python may return dict-like or pydantic model; handle both
    resp = r["response"] if isinstance(r, dict) else r.response
    return str(resp).strip()


# ----------------------------------------
# Task 2: text -> embedding numpy array
# ----------------------------------------
def compute_embedding(text: str, model: str = "embeddinggemma") -> np.ndarray:
    r = ollama.embeddings(model=model, prompt=text)
    emb = r["embedding"] if isinstance(r, dict) else r.embedding
    return np.array(emb, dtype=np.float32)


# ----------------------------------------
# Task 5: save/load dataframe as CSV
# embedding is stored as a python list string
# ----------------------------------------
def save_model_csv(df: pd.DataFrame, csv_path: str) -> None:
    df_to_save = df.copy()
    df_to_save["embedding"] = df_to_save["embedding"].apply(
        lambda x: repr(x.tolist()) if isinstance(x, np.ndarray) else repr(list(x))
    )
    Path(csv_path).parent.mkdir(parents=True, exist_ok=True)
    df_to_save.to_csv(csv_path, index=False)


def load_model_csv(csv_path: str) -> pd.DataFrame:
    if not os.path.exists(csv_path):
        return pd.DataFrame(columns=["filename", "description", "embedding"])

    df = pd.read_csv(csv_path)
    if df.empty:
        return pd.DataFrame(columns=["filename", "description", "embedding"])

    def parse_embedding(v):
        if isinstance(v, (list, np.ndarray)):
            return np.array(v, dtype=np.float32)
        # stored as string like "[0.1, 0.2, ...]"
        parsed = ast.literal_eval(v)
        return np.array(parsed, dtype=np.float32)

    df["embedding"] = df["embedding"].apply(parse_embedding)
    return df


# ----------------------------------------
# Update model with new results
# If filename already exists -> replace row
# else -> append row
# ----------------------------------------
def upsert_rows(df_model: pd.DataFrame, new_rows: list[dict]) -> pd.DataFrame:
    if df_model is None or df_model.empty:
        return pd.DataFrame(new_rows, columns=["filename", "description", "embedding"])

    df = df_model.copy()
    existing = set(df["filename"].astype(str).tolist())

    for row in new_rows:
        fn = row["filename"]
        if fn in existing:
            df.loc[df["filename"] == fn, ["description", "embedding"]] = [
                row["description"],
                row["embedding"],
            ]
        else:
            df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
            existing.add(fn)

    return df


# ----------------------------------------
# Read list of images from a text file
# (ignore blank lines and comments)
# ----------------------------------------
def read_image_list(txt_path: str) -> list[str]:
    images = []
    with open(txt_path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            images.append(s)
    return images


def main():
    parser = argparse.ArgumentParser(description="Task 9: batch pipeline for image->desc+embedding model update")
    parser.add_argument("--images", required=True, help="Text file with image filenames, one per line")
    parser.add_argument("--model", required=True, help="Path to model CSV (will be created if missing)")
    parser.add_argument("--image-dir", default=".", help="Base directory where image files live (default: current dir)")
    parser.add_argument("--vision-model", default="ministral-3:3b", help="Vision model for description")
    parser.add_argument("--embed-model", default="embeddinggemma", help="Embedding model")
    args = parser.parse_args()

    # sanity: check Ollama reachable
    try:
        _ = ollama.list()
    except Exception as e:
        raise SystemExit(
            "ERROR: Cannot connect to Ollama server. "
            "Make sure `ollama serve` is running on this node/session.\n"
            f"Details: {e}"
        )

    image_list = read_image_list(args.images)
    if len(image_list) == 0:
        raise SystemExit(f"ERROR: No images found in {args.images}")

    df_model = load_model_csv(args.model)

    new_rows = []
    for rel_path in image_list:
        img_path = os.path.join(args.image_dir, rel_path)
        if not os.path.exists(img_path):
            print(f"[SKIP] Missing file: {img_path}")
            continue

        try:
            desc = prompt_image(img_path, model=args.vision_model)
            emb = compute_embedding(desc, model=args.embed_model)
            new_rows.append({"filename": rel_path, "description": desc, "embedding": emb})
            print(f"[OK] {rel_path} -> desc_len={len(desc)} emb_shape={emb.shape}")
        except Exception as e:
            print(f"[FAIL] {rel_path}: {e}")

    df_model = upsert_rows(df_model, new_rows)
    save_model_csv(df_model, args.model)
    print(f"\nSaved model with {len(df_model)} rows to: {args.model}")


if __name__ == "__main__":
    main()
