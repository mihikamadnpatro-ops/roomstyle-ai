"""Shared feature extraction, used by the training scripts AND app.py.
One copy guarantees training and prediction use identical features.

Three feature sets (the app picks the right one from the saved model's input size):
  v1  : 48  RGB colour histogram (legacy)
  v2  : 91  RGB + HSV histograms, edge energy, brightness
  cnn : 1280 embedding from a pretrained MobileNetV2 CNN (transfer learning)

For the CNN, training uses PyTorch (extract_cnn_features.py) and exports the network
to ONNX; the deployed app runs the ONNX file with onnxruntime (small and fast, no PyTorch).
The image pre-processing below is plain NumPy and is shared by both sides.
"""
import os
import numpy as np
from PIL import Image

N_FEATURES_V1 = 48
N_FEATURES_V2 = 91
N_FEATURES_CNN = 1280
ONNX_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cnn_feature_extractor.onnx")


def _load(img):
    if not isinstance(img, Image.Image):
        img = Image.open(img)
    return img.convert("RGB")


def _hist(channel, bins, rng):
    h, _ = np.histogram(channel, bins=bins, range=rng)
    return h / max(h.sum(), 1)


# ----------------------------------------------------------------- colour features
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


# ----------------------------------------------------------------- CNN features
_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)   # ImageNet statistics
_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def cnn_preprocess(img):
    """PIL image or path -> float32 array of shape (1, 3, 224, 224).
    Resize shorter side to 256, centre-crop 224, scale to 0-1, normalise with ImageNet mean/std."""
    img = _load(img)
    w, h = img.size
    scale = 256 / min(w, h)
    img = img.resize((max(256, round(w * scale)), max(256, round(h * scale))), Image.BILINEAR)
    w, h = img.size
    left, top = (w - 224) // 2, (h - 224) // 2
    img = img.crop((left, top, left + 224, top + 224))
    a = np.asarray(img, dtype=np.float32) / 255.0
    a = (a - _MEAN) / _STD
    return np.ascontiguousarray(a.transpose(2, 0, 1)[None], dtype=np.float32)


_session = None


def extract_features_cnn(img):
    """Run the exported CNN (ONNX) and return a 1280-number embedding."""
    global _session
    if _session is None:
        if not os.path.exists(ONNX_PATH):
            raise FileNotFoundError(
                "cnn_feature_extractor.onnx not found. Run 'python extract_cnn_features.py' "
                "to create it, then commit it next to app.py.")
        import onnxruntime as ort
        _session = ort.InferenceSession(ONNX_PATH, providers=["CPUExecutionProvider"])
    x = cnn_preprocess(img)
    out = _session.run(None, {_session.get_inputs()[0].name: x})[0]
    return np.asarray(out, dtype=float).reshape(-1)


# ----------------------------------------------------------------- selection
extract_features = extract_features_v2  # default colour extractor


def get_extractor(n_features):
    """Return the extractor matching a trained model's expected input size."""
    if n_features == N_FEATURES_V1:
        return extract_features_v1
    if n_features == N_FEATURES_V2:
        return extract_features_v2
    if n_features == N_FEATURES_CNN:
        return extract_features_cnn
    raise ValueError(f"Model expects {n_features} features; no matching extractor.")
