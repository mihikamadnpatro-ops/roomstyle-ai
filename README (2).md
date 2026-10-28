# 🎨 RoomStyle AI

**Interior style classifier and paint colour recommender.** Upload a room photo; the app predicts its interior design style and suggests matching paint colours.

Built for *Foundations of Machine Learning* (NMIMS MPSTME). A pretrained **CNN (MobileNetV2, transfer learning)** turns each photo into 1,280 numbers; **PCA** and **LDA** then reduce them and an **MLP** classifies the style. Everything is compared against a logistic regression baseline and a colour-histogram model using cross-validation.

## How it works

1. **Data:** room photos from Unsplash, one folder per style (`raw_images/<style>/`).
2. **CNN features (`extract_cnn_features.py`):** MobileNetV2 pretrained on ImageNet (frozen, no training) reads each photo and outputs a 1,280-number embedding. The network is exported to ONNX so the deployed app does not need PyTorch.
3. **Models (`train_model.py`):** logistic regression, MLP, PCA(95% variance) + MLP, LDA + MLP, all as scikit-learn `Pipeline`s so scaling/PCA/LDA are fitted on training data only (no leakage).
4. **Evaluation:** stratified 80/20 split plus 5-fold cross-validation; the best model is chosen by CV accuracy, not test accuracy.
5. **App (`app.py`):** Streamlit UI that runs the ONNX CNN, then the saved pipeline, and shows the style, confidence, paint colours and per-style probabilities.

### CNN in simple words
A CNN slides small filters over an image to find edges, textures and shapes, then combines them into higher-level patterns (sofa, brick wall, plants). MobileNetV2 already learned this from about 1.3 million ImageNet photos. We reuse it as a fixed feature extractor ("transfer learning") because 319 images are too few to train a CNN from scratch. The colour-histogram model is kept for comparison.

## Styles

| Style | Label in code |
|---|---|
| Bohemian | `bohemain` (dataset folder spelling) |
| Industrial | `industrial` |
| Minimalist | `minimalist` |
| Traditional | `traditional` |

## Project structure

```
roomstyle-ai/
├── app.py                       # Streamlit app
├── features.py                  # Shared feature code: colour (v1, v2) and CNN pre-processing + ONNX runner
├── extract_cnn_features.py      # CNN embeddings -> X_cnn.npy, y_cnn.npy, cnn_feature_extractor.onnx
├── extract_features.py          # Colour features -> X_features.npy, y_labels.npy
├── train_model.py               # Compares LR / MLP / PCA+MLP / LDA+MLP, saves best pipeline
├── images.py                    # Image count per style folder
├── style_classifier.pkl         # Trained pipeline
├── label_encoder.pkl            # Style names <-> numbers
├── cnn_feature_extractor.onnx   # Created by extract_cnn_features.py (needed by the app for CNN models)
├── X_cnn.npy, y_cnn.npy         # CNN features (created by you)
├── X_features.npy, y_labels.npy # Colour features
├── reports/cnn, reports/colour  # Plots from train_model.py
├── requirements.txt             # Runtime deps (Render)
├── requirements-train.txt       # + matplotlib, torch, torchvision, onnx
├── render.yaml
└── .streamlit/config.toml
```

## Run it, step by step

```bash
pip install -r requirements-train.txt

python images.py raw_images                    # check counts
python extract_cnn_features.py raw_images      # needs internet once (downloads ~14 MB weights)
python train_model.py cnn                      # trains + plots, saves the model for the app
python train_model.py colour --no-save         # optional: colour-feature results for comparison

python -m streamlit run app.py
```

`extract_cnn_features.py` checks that the ONNX file gives the same numbers as PyTorch and stops if not.

## Deploy on Render

1. Make sure these are committed: `style_classifier.pkl`, `label_encoder.pkl`, **`cnn_feature_extractor.onnx`**, plus the code and `requirements.txt`.
2. Push to GitHub; Render redeploys automatically (Blueprint via `render.yaml`, or a Web Service with build `pip install -r requirements.txt` and start `streamlit run app.py --server.port $PORT --server.address 0.0.0.0`, env `PYTHON_VERSION=3.12.7`).
3. The app uses `onnxruntime` only, so the free plan is enough. First load after sleeping can take about a minute.

`scikit-learn` is pinned to the version used to train the pickles (1.8.0). If you train with another version, update the pin.

## Results

4 classes, 319 images (255 train / 64 test). Chance is 25%. CV = 5-fold stratified cross-validation on the training set.

**CNN features (1,280 numbers from MobileNetV2):**

| Model | CV accuracy | Test accuracy |
|---|---|---|
| Logistic Regression | 0.725 ± 0.066 | 0.656 |
| MLP (raw features) | 0.737 ± 0.090 | 0.734 |
| PCA (95% variance, 163 of 1,280 components) + MLP | 0.737 ± 0.040 | 0.672 |
| PCA + LDA (3 components) + MLP | 0.710 ± 0.047 | 0.625 |

**Colour-histogram features (48), for comparison:**

| Model | CV accuracy | Test accuracy |
|---|---|---|
| Logistic Regression | 0.439 ± 0.036 | 0.453 |
| MLP (raw) | 0.443 ± 0.081 | 0.516 |
| PCA (95%) + MLP | 0.408 ± 0.055 | 0.562 |
| LDA + MLP | 0.439 ± 0.020 | 0.562 |

**What the results show**

- The pretrained CNN features raise cross-validated accuracy from about 0.44 to about 0.72–0.74, a gain much larger than the spread between folds.
- PCA keeps 163 of 1,280 components with the same CV accuracy as the raw MLP (0.737), so the input is about 8 times smaller at no cost in accuracy.
- Differences between the four CNN-based models are within the fold-to-fold spread, so no single model is claimed to be best. The deployed model is the one with the highest CV score (MLP on raw CNN features).
- Test accuracy is close to CV accuracy, which suggests no leakage and little overfitting.

## Limitations

- Small dataset, so scores vary between splits.
- Style labels are subjective; real rooms often mix styles.
- The CNN is frozen (not fine-tuned), and the backbone was trained on ImageNet, not on interiors.
- Paint names are hand-mapped suggestions, not official brand shades.
- A "coastal" class was dropped for lack of images.

## Future work

Fine-tune the last CNN layers, add data augmentation, try an SVM baseline, map styles to real paint-brand catalogues, eigen-rooms (PCA on raw pixels).

## Dataset and licensing

Images from [Unsplash](https://unsplash.com) (free use under the Unsplash licence), used for training only and not redistributed.

## Author

**Mihika Patro**, B.Tech Data Science, NMIMS MPSTME, Mumbai
[LinkedIn](https://linkedin.com/in/mihikapatro-54525331a) · [GitHub](https://github.com/mihikamadnpatro-ops)
