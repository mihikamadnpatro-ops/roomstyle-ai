"""
Transfer learning step: turn every room photo into a 1280-number CNN embedding.

  * Uses MobileNetV2 pretrained on ImageNet (weights are downloaded automatically the first
    time, about 14 MB, so you need internet once).
  * The CNN is FROZEN (no training): we only read its output for each image.
  * Saves  X_cnn.npy (n x 1280)  and  y_cnn.npy (labels).
  * Exports the CNN to cnn_feature_extractor.onnx so the deployed app does not need PyTorch,
    and checks that the ONNX output matches PyTorch.

Usage:  python extract_cnn_features.py [path/to/raw_images]
Folder layout: raw_images/<style_name>/*.jpg
"""
import os
import sys
import numpy as np
import torch
import torch.nn as nn
from torchvision import models
from features import cnn_preprocess, N_FEATURES_CNN, ONNX_PATH

data_dir = sys.argv[1] if len(sys.argv) > 1 else "raw_images"
BATCH = 32
EXTS = (".jpg", ".jpeg", ".png", ".webp", ".bmp")


class Embedder(nn.Module):
    """MobileNetV2 convolutional body + global average pooling -> 1280 numbers."""
    def __init__(self):
        super().__init__()
        weights = models.MobileNet_V2_Weights.IMAGENET1K_V1
        self.features = models.mobilenet_v2(weights=weights).features

    def forward(self, x):
        return self.features(x).mean(dim=[2, 3])


model = Embedder().eval()
for p in model.parameters():
    p.requires_grad = False

# ---- collect files
files, labels = [], []
for style in sorted(os.listdir(data_dir)):
    sp = os.path.join(data_dir, style)
    if not os.path.isdir(sp):
        continue
    for fname in sorted(os.listdir(sp)):
        if fname.lower().endswith(EXTS):
            files.append(os.path.join(sp, fname))
            labels.append(style)
print(f"Found {len(files)} images in {len(set(labels))} styles")

# ---- extract embeddings in batches
X, y = [], []
batch, batch_labels = [], []


def flush():
    global batch, batch_labels
    if not batch:
        return
    with torch.no_grad():
        emb = model(torch.from_numpy(np.concatenate(batch))).numpy()
    X.append(emb)
    y.extend(batch_labels)
    batch, batch_labels = [], []


for i, (f, lab) in enumerate(zip(files, labels), 1):
    try:
        batch.append(cnn_preprocess(f))
        batch_labels.append(lab)
    except Exception as e:
        print(f"Skipped {f}: {e}")
    if len(batch) == BATCH:
        flush()
        print(f"  {i}/{len(files)} done")
flush()

X = np.concatenate(X)
y = np.array(y)
assert X.shape[1] == N_FEATURES_CNN, X.shape
print("CNN feature matrix:", X.shape)
np.save("X_cnn.npy", X)
np.save("y_cnn.npy", y)

# ---- export to ONNX (used by the deployed app)
dummy = torch.zeros(1, 3, 224, 224)
kwargs = dict(input_names=["input"], output_names=["embedding"],
              dynamic_axes={"input": {0: "batch"}, "embedding": {0: "batch"}}, opset_version=17)
try:
    torch.onnx.export(model, dummy, ONNX_PATH, dynamo=False, **kwargs)
except TypeError:                      # older torch has no 'dynamo' argument
    torch.onnx.export(model, dummy, ONNX_PATH, **kwargs)
print("Saved", ONNX_PATH, f"({os.path.getsize(ONNX_PATH) / 1e6:.1f} MB)")

# ---- verify ONNX == PyTorch on a real image
import onnxruntime as ort
sess = ort.InferenceSession(ONNX_PATH, providers=["CPUExecutionProvider"])
x = cnn_preprocess(files[0])
with torch.no_grad():
    ref = model(torch.from_numpy(x)).numpy()
out = sess.run(None, {"input": x})[0]
diff = float(np.abs(ref - out).max())
print(f"Max difference PyTorch vs ONNX: {diff:.2e}")
if diff > 1e-3:
    raise SystemExit("ONNX output does not match PyTorch - do not deploy this file.")
print("OK. Next: python train_model.py")
