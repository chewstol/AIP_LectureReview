from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


DEFAULT_PERSONAS_PATH = Path("data/generated_personas.json")
DEFAULT_NODES_PATH = Path("data/model/lecture_nodes_with_text.csv")
DEFAULT_OUT_DIR = Path("data/persona_weights")


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def load_node_columns(path: Path) -> set[str]:
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        reader = csv.reader(file)
        try:
            return set(next(reader))
        except StopIteration as exc:
            raise ValueError(f"Nodes CSV is empty: {path}") from exc


def ensure_persona_list(value: Any, path: Path) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        raise ValueError(f"{path} must contain a JSON array of persona objects.")
    personas = []
    for idx, item in enumerate(value, start=1):
        if not isinstance(item, dict):
            raise ValueError(f"Persona at index {idx} is not a JSON object.")
        personas.append(item)
    return personas


def numeric_vector(persona: dict[str, Any], vector_field: str, node_columns: set[str]) -> dict[str, float]:
    vector = persona.get(vector_field)
    persona_id = persona.get("persona_id", "<unknown>")
    if not isinstance(vector, dict):
        raise ValueError(f"Persona {persona_id} has no object field `{vector_field}`.")

    converted: dict[str, float] = {}
    invalid = []
    unknown = []
    for key, value in vector.items():
        feature = str(key)
        if feature not in node_columns:
            unknown.append(feature)
            continue
        try:
            converted[feature] = float(value)
        except (TypeError, ValueError):
            invalid.append(feature)

    if unknown:
        raise ValueError(f"Persona {persona_id} has feature(s) not found in nodes CSV: {sorted(unknown)}")
    if invalid:
        raise ValueError(f"Persona {persona_id} has non-numeric feature value(s): {sorted(invalid)}")
    if not converted:
        raise ValueError(f"Persona {persona_id} produced an empty vector from `{vector_field}`.")
    return converted


def safe_name(value: Any) -> str:
    text = str(value or "persona").strip().lower()
    chars = [char if char.isalnum() else "_" for char in text]
    return "_".join("".join(chars).split("_")).strip("_") or "persona"


def write_persona_weights(
    personas: list[dict[str, Any]],
    node_columns: set[str],
    out_dir: Path,
    vector_field: str,
) -> list[dict[str, Any]]:
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_rows: list[dict[str, Any]] = []

    for index, persona in enumerate(personas, start=1):
        persona_id = persona.get("persona_id", index)
        preset_name = safe_name(persona.get("preset_name", "persona"))
        weights = numeric_vector(persona, vector_field, node_columns)
        stem = f"persona_{int(persona_id):03d}_{preset_name}_{safe_name(vector_field)}"
        weights_path = out_dir / f"{stem}.json"
        weights_path.write_text(json.dumps(weights, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

        manifest_rows.append(
            {
                "persona_id": persona_id,
                "preset_name": persona.get("preset_name", ""),
                "vector_field": vector_field,
                "feature_count": len(weights),
                "weights_file": str(weights_path.as_posix()),
                "top_k": persona.get("top_k", ""),
                "sample_n": persona.get("sample_n", ""),
                "selected_review_count": len(persona.get("selected_reviews", []))
                if isinstance(persona.get("selected_reviews"), list)
                else "",
            }
        )

    manifest_path = out_dir / f"manifest_{safe_name(vector_field)}.csv"
    with manifest_path.open("w", encoding="utf-8-sig", newline="") as file:
        fieldnames = list(manifest_rows[0].keys()) if manifest_rows else []
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(manifest_rows)

    return manifest_rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert generated persona vectors into recommend_topk.py --weights-file compatible JSON files."
    )
    parser.add_argument("--personas", type=Path, default=DEFAULT_PERSONAS_PATH)
    parser.add_argument("--nodes", type=Path, default=DEFAULT_NODES_PATH)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument(
        "--vector-field",
        choices=["initial_preference_vector", "aggregated_review_vector"],
        default="initial_preference_vector",
        help="Persona vector field to export as feature weights.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    personas = ensure_persona_list(load_json(args.personas), args.personas)
    node_columns = load_node_columns(args.nodes)
    manifest_rows = write_persona_weights(
        personas=personas,
        node_columns=node_columns,
        out_dir=args.out_dir,
        vector_field=args.vector_field,
    )

    manifest_path = args.out_dir / f"manifest_{safe_name(args.vector_field)}.csv"
    print(f"[OK] personas: {len(personas)}")
    print(f"[OK] vector field: {args.vector_field}")
    print(f"[OK] weight files: {args.out_dir}")
    print(f"[OK] manifest: {manifest_path}")
    if manifest_rows:
        print(f"[OK] example weights file: {manifest_rows[0]['weights_file']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
