from __future__ import annotations

"""Compare quality models through their effect on the Top-K recommendation.

Holds the preference (a persona weights file or a preset) FIXED and swaps only
the quality model that feeds ``predicted_quality``. Reports, per model: the
in-sample fit (RMSE vs the actual rating), the resulting Top-K lectures, the
average rating of those lectures, and how much each model's list overlaps the
others (pairwise Jaccard). This turns the offline model comparison into a
"so what does it change in the actual recommendations" view for the report.

numpy/pandas only (KoBERT models reuse the cached embeddings). Example:

    python scripts/compare_quality_topk.py --weights-file data/personas_ac/persona_002.json --score-mode objective
    python scripts/compare_quality_topk.py --preset low_workload --score-mode similarity
"""

import argparse
import json
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

from build_personas import PERSONA_FEATURE_COLUMNS, TEXT_FEATURES, load_nodes_16
from quality_models import QUALITY_MODELS, predict_quality
from recommend_topk import PRESETS
from topk_engine import TARGET_COLUMN, make_recommendations, select_topk


def load_preference(args: argparse.Namespace) -> tuple[str, dict[str, float]]:
    if args.weights_file:
        weights = json.loads(args.weights_file.read_text(encoding="utf-8"))
        return args.weights_file.stem, {str(k): float(v) for k, v in weights.items()}
    return args.preset, dict(PRESETS[args.preset])


def load_subject_names(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    subjects = pd.read_csv(path, encoding="utf-8-sig", dtype=str).drop_duplicates("lectureId")
    return {
        str(row["lectureId"]): f"{row.get('name', '?')}({row.get('professor', '')})"
        for _, row in subjects.iterrows()
    }


def jaccard(a: list[str], b: list[str]) -> float:
    sa, sb = set(a), set(b)
    return len(sa & sb) / len(sa | sb) if sa | sb else 0.0


def main() -> None:
    args = parse_args()
    nodes = load_nodes_16(args.nodes)
    names = load_subject_names(args.subjects)
    scenario, weights = load_preference(args)

    # In similarity mode the preference vector must only reference known feature
    # columns; drop any preset keys outside the 16-feature schema.
    if args.score_mode == "similarity":
        weights = {k: v for k, v in weights.items() if k in PERSONA_FEATURE_COLUMNS}

    y = nodes[TARGET_COLUMN].to_numpy(dtype=float)
    models = args.models or QUALITY_MODELS
    topk_by_model: dict[str, pd.DataFrame] = {}
    rows = []

    for model in models:
        quality = predict_quality(model, nodes, alpha=args.alpha, text_columns=TEXT_FEATURES)
        rmse = float(np.sqrt(np.mean((quality - y) ** 2)))
        recommendations = make_recommendations(
            nodes=nodes,
            feature_columns=PERSONA_FEATURE_COLUMNS,
            preference_weights=weights,
            similarity_weight=args.similarity_weight,
            quality_weight=args.quality_weight,
            alpha=args.alpha,
            score_mode=args.score_mode,
            valence=args.valence,
            quality_override=quality,
        )
        topk = select_topk(recommendations, args.top_k, feature_columns=PERSONA_FEATURE_COLUMNS)
        topk_by_model[model] = topk
        rows.append(
            {
                "quality_model": model,
                "insample_rmse": round(rmse, 5),
                "topk_avg_rating": round(float(topk["rating_average"].mean()), 3),
                "top5": " | ".join(names.get(str(l), str(l)) for l in topk["lecture_id"].head(5)),
            }
        )

    summary = pd.DataFrame(rows)
    print(f"\n=== Quality-model comparison in Top-{args.top_k}  (scenario: {scenario}, score_mode: {args.score_mode}) ===\n")
    print(summary[["quality_model", "insample_rmse", "topk_avg_rating"]].to_string(index=False))

    print("\n--- pairwise Top-K overlap (Jaccard; low = the model changed the recommendations) ---")
    lists = {m: topk_by_model[m]["lecture_id"].astype(str).tolist() for m in models}
    base = models[0]
    for model in models[1:]:
        print(f"  {base} vs {model}: {jaccard(lists[base], lists[model]):.2f}")
    if len(models) > 2:
        allpairs = np.mean([jaccard(lists[a], lists[b]) for a, b in combinations(models, 2)])
        print(f"  mean over all pairs: {allpairs:.2f}")

    print("\n--- Top-5 per model ---")
    for _, r in summary.iterrows():
        print(f"  [{r['quality_model']:<17}] {r['top5']}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    side_by_side = pd.DataFrame(
        {m: [names.get(str(l), str(l)) for l in topk_by_model[m]["lecture_id"].head(args.top_k)] for m in models}
    )
    side_by_side.insert(0, "rank", np.arange(1, len(side_by_side) + 1))
    side_by_side.to_csv(args.out, index=False, encoding="utf-8-sig")
    summary.to_csv(args.out.with_name(args.out.stem + "_summary.csv"), index=False, encoding="utf-8-sig")
    print(f"\n[OK] wrote side-by-side: {args.out}")
    print(f"[OK] wrote summary: {args.out.with_name(args.out.stem + '_summary.csv')}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare quality models by their effect on Top-K recommendations.")
    parser.add_argument("--weights-file", type=Path, help="Persona weights json (overrides --preset).")
    parser.add_argument("--preset", choices=sorted(PRESETS), default="low_workload")
    parser.add_argument("--score-mode", choices=["similarity", "objective", "objective_rel"], default="objective")
    parser.add_argument("--valence", choices=["avoid", "seek"], default="avoid")
    parser.add_argument("--models", nargs="*", choices=QUALITY_MODELS, help="Subset of models (default: all).")
    parser.add_argument("--nodes", type=Path, default=Path("data/model/lecture_nodes_with_text.csv"))
    parser.add_argument("--subjects", type=Path, default=Path("subjects.csv"))
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--similarity-weight", type=float, default=0.7)
    parser.add_argument("--quality-weight", type=float, default=0.3)
    parser.add_argument("--alpha", type=float, default=10.0)
    parser.add_argument("--out", type=Path, default=Path("data/recommendations/quality_model_compare.csv"))
    return parser.parse_args()


if __name__ == "__main__":
    main()
