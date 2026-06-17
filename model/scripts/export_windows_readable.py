from __future__ import annotations

import argparse
import csv
from pathlib import Path


def convert_csv(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    with src.open("r", encoding="utf-8-sig", newline="") as input_file:
        reader = csv.reader(input_file)
        with dst.open("w", encoding="cp949", errors="replace", newline="") as output_file:
            writer = csv.writer(output_file)
            writer.writerows(reader)


def convert_text(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    text = src.read_text(encoding="utf-8-sig")
    dst.write_text(text, encoding="utf-8-sig")


def main() -> int:
    parser = argparse.ArgumentParser(description="Export UTF-8 files to Windows-readable copies.")
    parser.add_argument("--root", type=Path, default=Path("data"))
    parser.add_argument("--out", type=Path, default=Path("data/windows_readable"))
    args = parser.parse_args()

    converted = 0
    for src in sorted(args.root.rglob("*")):
        if args.out in src.parents:
            continue
        if src.suffix.lower() == ".csv":
            dst = args.out / src.relative_to(args.root)
            convert_csv(src, dst)
            converted += 1
        elif src.suffix.lower() in {".md", ".txt"}:
            dst = args.out / src.relative_to(args.root)
            convert_text(src, dst)
            converted += 1

    print(f"[OK] converted files: {converted}")
    print(f"[OK] output: {args.out.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
