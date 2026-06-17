from __future__ import annotations

import argparse
import re
import zipfile
from pathlib import Path
from xml.etree import ElementTree as ET


NS = {
    "a": "http://schemas.openxmlformats.org/drawingml/2006/main",
    "p": "http://schemas.openxmlformats.org/presentationml/2006/main",
}


def slide_sort_key(name: str) -> int:
    match = re.search(r"slide(\d+)\.xml$", name)
    return int(match.group(1)) if match else 0


def extract_text_from_slide(xml: bytes) -> list[str]:
    root = ET.fromstring(xml)
    paragraphs: list[str] = []
    for paragraph in root.findall(".//a:p", NS):
        texts = [node.text or "" for node in paragraph.findall(".//a:t", NS)]
        text = "".join(texts).strip()
        if text:
            paragraphs.append(text)
    return paragraphs


def extract_notes_from_slide(zip_file: zipfile.ZipFile, slide_index: int) -> list[str]:
    note_name = f"ppt/notesSlides/notesSlide{slide_index}.xml"
    if note_name not in zip_file.namelist():
        return []
    return extract_text_from_slide(zip_file.read(note_name))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("pptx", type=Path)
    parser.add_argument("--out", type=Path, default=Path("data/ppt_review/slide_text.md"))
    args = parser.parse_args()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(args.pptx) as zf:
        slide_names = sorted(
            [name for name in zf.namelist() if re.search(r"ppt/slides/slide\d+\.xml$", name)],
            key=slide_sort_key,
        )
        lines = [f"# PPTX Text Extract", "", f"File: `{args.pptx}`", "", f"Slides: {len(slide_names)}", ""]
        for idx, slide_name in enumerate(slide_names, start=1):
            lines.append(f"## Slide {idx}")
            texts = extract_text_from_slide(zf.read(slide_name))
            if texts:
                for text in texts:
                    lines.append(f"- {text}")
            else:
                lines.append("- (no text)")
            notes = extract_notes_from_slide(zf, idx)
            if notes:
                lines.append("")
                lines.append("Notes:")
                for note in notes:
                    lines.append(f"- {note}")
            lines.append("")
    args.out.write_text("\n".join(lines), encoding="utf-8")
    print(f"[OK] wrote {args.out.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
