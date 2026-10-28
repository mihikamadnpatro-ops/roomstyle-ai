"""Turn raw images into a feature matrix.
Usage:  python extract_features.py [path/to/raw_images]
Folder layout: raw_images/<style_name>/*.jpg
"""
import os
import sys
import numpy as np
from features import extract_features

data_dir = sys.argv[1] if len(sys.argv) > 1 else "raw_images"

X, y = [], []
for style in sorted(os.listdir(data_dir)):
    style_path = os.path.join(data_dir, style)
    if not os.path.isdir(style_path):
        continue
    for fname in os.listdir(style_path):
        fpath = os.path.join(style_path, fname)
        try:
            X.append(extract_features(fpath))
            y.append(style)
        except Exception as e:
            print(f"Skipped {fpath}: {e}")

X, y = np.array(X), np.array(y)
print("Feature matrix shape:", X.shape)
np.save("X_features.npy", X)
np.save("y_labels.npy", y)
