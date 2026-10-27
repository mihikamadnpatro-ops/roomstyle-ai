"""Quick check of image counts per style folder.
Usage:  python images.py [path/to/raw_images]
"""
import os
import sys

data_dir = sys.argv[1] if len(sys.argv) > 1 else "raw_images"
for style in sorted(os.listdir(data_dir)):
    p = os.path.join(data_dir, style)
    if os.path.isdir(p):
        print(f"{style}: {len(os.listdir(p))} images")
