"""Shared feature extraction, used by extract_features.py AND app.py.
One copy guarantees training and prediction use identical features.

v2 (91 features, default for new training):
  RGB hist 16x3 = 48 | HSV hist H16+S8+V8 = 32 | 3x3 edge energy = 9 | brightness mean/std = 2
v1 (48 features, legacy): RGB hist only, on 64x64 images.

The app picks the right extractor automatically from the saved model's input size.
"""
import numpy as np
from PIL import Image

N_FEATURES_V1 = 48
N_FEATURES_V2 = 91


def _load(img):
    if not isinstance(img, Image.Image):
        img = Image.open(img)
    return img.convert("RGB")


def _hist(channel, bins, rng):
    h, _ = np.histogram(channel, bins=bins, range=rng)
    return h / max(h.sum(), 1)


def extract_features_v1(img, size=(64, 64)):
    rgb = np.array(_load(img).resize(size))
    f = np.concatenate([np.histogram(rgb[:, :, c], bins=16, range=(0, 256))[0]
                        for c in range(3)]).astype(float)
    return f / f.sum()


def extract_features_v2(img, size=(128, 128)):
    img = _load(img).resize(size)
    rgb = np.array(img)
    f_rgb = np.concatenate([_hist(rgb[:, :, c], 16, (0, 256)) for c in range(3)])
    hsv = np.array(img.convert("HSV"))
    f_hsv = np.concatenate([_hist(hsv[:, :, 0], 16, (0, 256)),
                            _hist(hsv[:, :, 1], 8, (0, 256)),
                            _hist(hsv[:, :, 2], 8, (0, 256))])
    gray = np.array(img.convert("L")).astype(float)
    gy, gx = np.gradient(gray)
    mag = np.hypot(gx, gy)
    h, w = mag.shape
    f_tex = np.array([mag[i * h // 3:(i + 1) * h // 3, j * w // 3:(j + 1) * w // 3].mean()
                      for i in range(3) for j in range(3)]) / 255.0
    f_br = np.array([gray.mean() / 255.0, gray.std() / 255.0])
    return np.concatenate([f_rgb, f_hsv, f_tex, f_br]).astype(float)


extract_features = extract_features_v2  # default for new training


def get_extractor(n_features):
    """Return the extractor matching a trained model's expected input size."""
    if n_features == N_FEATURES_V1:
        return extract_features_v1
    if n_features == N_FEATURES_V2:
        return extract_features_v2
    raise ValueError(f"Model expects {n_features} features; no matching extractor.")
