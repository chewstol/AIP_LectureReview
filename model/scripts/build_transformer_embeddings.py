from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from transformers import AutoModel, AutoTokenizer


def mean_pool(last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    mask = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
    summed = torch.sum(last_hidden_state * mask, dim=1)
    counts = torch.clamp(mask.sum(dim=1), min=1e-9)
    return summed / counts


def main() -> int:
    parser = argparse.ArgumentParser(description="Build lecture-level transformer embeddings from review text.")
    parser.add_argument("--articles", type=Path, default=Path("data/normalized/lecture_articles.csv"))
    parser.add_argument("--nodes", type=Path, default=Path("data/model/lecture_nodes_with_text.csv"))
    parser.add_argument("--model", default="skt/kobert-base-v1")
    parser.add_argument("--output", type=Path, default=Path("data/model/lecture_kobert_embeddings.npz"))
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-length", type=int, default=192)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    args = parser.parse_args()

    nodes = pd.read_csv(args.nodes, usecols=["lecture_id"])
    articles = pd.read_csv(args.articles, usecols=["lecture_id", "text"])
    valid_ids = set(nodes["lecture_id"].astype(str))
    articles["lecture_id"] = articles["lecture_id"].astype(str)
    articles["text"] = articles["text"].fillna("").astype(str).str.strip()
    articles = articles[
        articles["lecture_id"].isin(valid_ids) & articles["text"].ne("")
    ].reset_index(drop=True)

    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device

    print(f"[INFO] model={args.model}")
    print(f"[INFO] device={device}")
    print(f"[INFO] reviews={len(articles)}, lectures={articles['lecture_id'].nunique()}")

    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=False)
    model = AutoModel.from_pretrained(args.model).to(device)
    model.eval()

    sums: dict[str, np.ndarray] = {}
    counts: dict[str, int] = {}
    texts = articles["text"].tolist()
    lecture_ids = articles["lecture_id"].tolist()

    with torch.inference_mode():
        for start in range(0, len(texts), args.batch_size):
            batch_texts = texts[start : start + args.batch_size]
            batch_ids = lecture_ids[start : start + args.batch_size]
            encoded = tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=args.max_length,
                return_tensors="pt",
            )
            encoded = {key: value.to(device) for key, value in encoded.items()}
            output = model(**encoded)
            pooled = mean_pool(output.last_hidden_state, encoded["attention_mask"])
            embeddings = pooled.cpu().numpy().astype(np.float32)

            for lecture_id, embedding in zip(batch_ids, embeddings):
                if lecture_id not in sums:
                    sums[lecture_id] = embedding.copy()
                    counts[lecture_id] = 1
                else:
                    sums[lecture_id] += embedding
                    counts[lecture_id] += 1

            completed = min(start + args.batch_size, len(texts))
            if completed % 320 == 0 or completed == len(texts):
                print(f"[PROGRESS] {completed}/{len(texts)} reviews")

    ordered_ids = nodes["lecture_id"].astype(str).tolist()
    missing = [lecture_id for lecture_id in ordered_ids if lecture_id not in sums]
    if missing:
        raise ValueError(f"Missing embeddings for {len(missing)} lectures: {missing[:10]}")

    matrix = np.stack([sums[lecture_id] / counts[lecture_id] for lecture_id in ordered_ids])
    review_counts = np.array([counts[lecture_id] for lecture_id in ordered_ids], dtype=np.int32)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        lecture_id=np.array(ordered_ids),
        embeddings=matrix,
        review_count=review_counts,
        model=np.array([args.model]),
        max_length=np.array([args.max_length]),
    )
    print(f"[OK] embedding shape={matrix.shape}")
    print(f"[OK] output={args.output.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
