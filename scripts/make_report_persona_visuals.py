"""Generate persona statistics figures for the final report.

Reads the 10-persona diverse set in data/personas_test10 and emits two figures:
  1. persona_weight_heatmap.png — each persona's 16-dim preference vector.
  2. persona_valence_poles.png   — mean friendly-pole vs burden-pole emphasis,
     split by valence (avoid / seek), supporting the report's validation claim.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager, rcParams

rcParams["font.family"] = "Malgun Gothic"
rcParams["axes.unicode_minus"] = False

ROOT = Path(__file__).resolve().parents[1]
PERSONA_DIR = ROOT / "data" / "personas_test10"
OUT = ROOT / "report_assets"
OUT.mkdir(exist_ok=True)

FEATURES = [
    "assignment_low_score", "assignment_high_score",
    "teamwork_low_score", "teamwork_high_score",
    "grading_generous_score", "grading_strict_score",
    "attendance_light_score", "attendance_strict_score",
    "exam_light_score", "exam_heavy_score",
    "text_assignment_tfidf", "text_exam_tfidf", "text_teamwork_tfidf",
    "text_attendance_tfidf", "text_grading_tfidf", "text_teaching_tfidf",
]
FRIENDLY = ["assignment_low_score", "teamwork_low_score", "grading_generous_score",
            "attendance_light_score", "exam_light_score"]
BURDEN = ["assignment_high_score", "teamwork_high_score", "grading_strict_score",
          "attendance_strict_score", "exam_heavy_score"]

metas = sorted(PERSONA_DIR.glob("persona_*_meta.json"))
rows, labels, valences = [], [], []
for path in metas:
    meta = json.loads(path.read_text(encoding="utf-8"))
    weights = meta["weights"]
    rows.append([float(weights[f]) for f in FEATURES])
    tgt = meta.get("target_category", "")
    val = meta.get("valence", "")
    valences.append(val)
    labels.append(f"{meta['persona'].replace('persona_','P')} · {tgt}/{val}")

matrix = np.array(rows)

# ----- Figure 1: weight heatmap -----------------------------------------
fig, ax = plt.subplots(figsize=(11, 6))
im = ax.imshow(matrix, aspect="auto", cmap="YlGnBu", vmin=0, vmax=1)
ax.set_xticks(range(len(FEATURES)))
ax.set_xticklabels(FEATURES, rotation=55, ha="right", fontsize=8)
ax.set_yticks(range(len(labels)))
ax.set_yticklabels(labels, fontsize=9)
ax.axvline(9.5, color="white", lw=2)
ax.set_title("페르소나별 16차원 선호 벡터 (좌: 구조화 10 · 우: 텍스트 6)", fontsize=12, pad=12)
cbar = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
cbar.set_label("weight", fontsize=9)
fig.tight_layout()
fig.savefig(OUT / "persona_weight_heatmap.png", dpi=150)
plt.close(fig)

# ----- Figure 2: on-target vs off-target text salience ------------------
# Demonstrates the category-biased sampling: each persona's text signal
# concentrates on its assigned target axis vs the average of the other axes.
CATS = ["assignment", "exam", "teamwork", "attendance", "grading", "teaching"]
on_vals, off_vals, plabels = [], [], []
for path in metas:
    meta = json.loads(path.read_text(encoding="utf-8"))
    w = meta["weights"]
    tgt = meta["target_category"]
    vals = {c: float(w[f"text_{c}_tfidf"]) for c in CATS}
    on_vals.append(vals[tgt])
    off_vals.append(np.mean([vals[c] for c in CATS if c != tgt]))
    plabels.append(f"{meta['persona'].replace('persona_','P')}\n{tgt}")

x = np.arange(len(plabels))
w = 0.38
fig, ax = plt.subplots(figsize=(11, 5))
ax.bar(x - w/2, on_vals, w, label="타깃 축 텍스트 언급도", color="#4C72B0")
ax.bar(x + w/2, off_vals, w, label="비타깃 축 평균 언급도", color="#C9C9C9")
ax.axhline(np.mean(on_vals), color="#4C72B0", ls="--", lw=1,
           label=f"타깃 평균 {np.mean(on_vals):.2f}")
ax.axhline(np.mean(off_vals), color="#888888", ls="--", lw=1,
           label=f"비타깃 평균 {np.mean(off_vals):.2f}")
ax.set_xticks(x)
ax.set_xticklabels(plabels, fontsize=9)
ax.set_ylabel("text_{축}_tfidf (강의 스케일 정규화)", fontsize=10)
ax.set_title("페르소나별 타깃 축 집중도 (카테고리 편향 샘플링 효과)", fontsize=12, pad=10)
ax.legend(fontsize=9, ncol=2)
fig.tight_layout()
fig.savefig(OUT / "persona_target_salience.png", dpi=150)
plt.close(fig)

print("wrote:", OUT / "persona_weight_heatmap.png")
print("wrote:", OUT / "persona_target_salience.png")
