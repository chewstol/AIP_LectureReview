from __future__ import annotations

from pathlib import Path


SRC = Path("data/ppt_review/slide_text.md")
DST = Path("data/ppt_review/slide_text_repaired_best_effort.md")


def repair(text: str) -> str:
    try:
        return text.encode("cp949", errors="replace").decode("utf-8", errors="replace")
    except UnicodeError:
        return text


def main() -> int:
    text = SRC.read_text(encoding="utf-8")
    DST.write_text(repair(text), encoding="utf-8")
    print(f"[OK] wrote {DST.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
