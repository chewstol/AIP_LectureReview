from __future__ import annotations

import re
import zipfile
from pathlib import Path
from xml.etree import ElementTree as ET

from PIL import Image


PPTX = Path(r"C:\Users\jungseobum\Documents\카카오톡 받은 파일\AIP_중간발표_1회차.pptx")


def main() -> int:
    with zipfile.ZipFile(PPTX) as zf:
        print("Media")
        for name in sorted(n for n in zf.namelist() if n.startswith("ppt/media/")):
            if name.lower().endswith((".png", ".jpg", ".jpeg")):
                with zf.open(name) as file:
                    image = Image.open(file)
                    print(name, image.size)
            else:
                print(name, "svg/vector")

        print("\nSlide relationships")
        for idx in range(1, 11):
            rel_name = f"ppt/slides/_rels/slide{idx}.xml.rels"
            if rel_name not in zf.namelist():
                continue
            xml = zf.read(rel_name).decode("utf-8", errors="ignore")
            targets = re.findall(r'Target="([^"]+)"', xml)
            print(f"slide {idx}: {targets}")

        print("\nSlide picture counts")
        ns = {
            "p": "http://schemas.openxmlformats.org/presentationml/2006/main",
        }
        for idx in range(1, 11):
            slide_name = f"ppt/slides/slide{idx}.xml"
            root = ET.fromstring(zf.read(slide_name))
            pics = root.findall(".//p:pic", ns)
            print(f"slide {idx}: {len(pics)} picture(s)")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
