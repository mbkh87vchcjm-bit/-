#!/usr/bin/env python3
"""Builds the small, committed test fixtures used by the JVM/golden tests.

Source images (all public photographs of public figures, used strictly for
academic evaluation of the matcher):

  * examples/obama.jpg, obama2.jpg, biden.jpg, biden2.jpg, two_people.jpg,
    kit_with_rose.jpg — from github.com/ageitgey/face_recognition (MIT repo)
  * LFWCrop crops — aligned 250x250 crops of LFW

Nothing is downloaded here: pass the directories with --fr and --lfw.
Outputs are resized to <= 640 px and saved as JPEG (quality 90) to keep the
repository small.
"""
from __future__ import annotations

import argparse
import os

import numpy as np
from PIL import Image

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")


def save(src_path: str, name: str, max_side: int = 640) -> None:
    im = Image.open(src_path).convert("RGB")
    w, h = im.size
    scale = min(1.0, max_side / max(w, h))
    if scale < 1.0:
        im = im.resize((int(round(w * scale)), int(round(h * scale))), Image.BILINEAR)
    im.save(os.path.join(OUT, name), quality=90)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fr", required=True, help="path to the face_recognition examples dir")
    ap.add_argument("--lfw", required=True, help="path to LFW faces dir")
    args = ap.parse_args()
    os.makedirs(os.path.join(OUT, "scenes"), exist_ok=True)
    os.makedirs(os.path.join(OUT, "lfw"), exist_ok=True)

    scenes = {
        "obama.jpg": "obama.jpg",
        "obama2.jpg": "obama2.jpg",
        "biden.jpg": "biden.jpg",
        "biden2.jpg": "knn_examples/train/biden/biden2.jpg",
        "two_people.jpg": "two_people.jpg",
        "kit_and_rose.jpg": "knn_examples/test/kit_with_rose.jpg",
        "obama_partial_face.jpg": "../tests/test_images/obama_partial_face.jpg",
        "johnsnow.jpg": "knn_examples/test/johnsnow_test1.jpg",
    }
    for dst, src in scenes.items():
        path = os.path.join(args.fr, "examples", src)
        if not os.path.exists(path):
            path = os.path.join(args.fr, "tests", "test_images", os.path.basename(src))
        save(path, os.path.join("scenes", dst))
        print("scene", dst, Image.open(os.path.join(OUT, "scenes", dst)).size)

    lfw_pick = [
        "Aaron_Peirsol_0001", "Aaron_Peirsol_0002", "Aaron_Peirsol_0003",
        "AJ_Cook_0001", "AJ_Lamas_0001", "Aaron_Eckhart_0001",
        "Aaron_Guiel_0001", "Aaron_Patterson_0001",
    ]
    for name in lfw_pick:
        src = os.path.join(args.lfw, name + ".ppm")
        if not os.path.exists(src):
            print("missing", src)
            continue
        save(src, os.path.join("lfw", name + ".jpg"), max_side=250)
        print("lfw", name)

    # A deliberately unusable image (uniform) to test the validation gate.
    Image.fromarray(np.full((320, 240, 3), 128, dtype=np.uint8)).save(
        os.path.join(OUT, "scenes", "blank.jpg"), quality=90)
    print("done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
