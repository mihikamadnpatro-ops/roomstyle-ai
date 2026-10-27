import numpy as np
import streamlit as st
import joblib
from PIL import Image
from features import get_extractor

st.set_page_config(page_title="RoomStyle AI", page_icon="🎨")


@st.cache_resource
def load_artifacts():
    # style_classifier.pkl is a full Pipeline: scaler -> (PCA/LDA) -> classifier
    model = joblib.load("style_classifier.pkl")
    le = joblib.load("label_encoder.pkl")
    extractor = get_extractor(model.named_steps["scaler"].n_features_in_)
    return model, le, extractor


model, le, extractor = load_artifacts()

color_recommendations = {
    "bohemain": ["Terracotta Warmth", "Desert Sand", "Deep Mustard"],
    "industrial": ["Charcoal Grey", "Concrete Grey", "Steel Blue"],
    "minimalist": ["Pure White", "Soft Greige", "Pale Stone"],
    "traditional": ["Deep Burgundy", "Forest Green", "Warm Ivory"],
}

st.title("🎨 RoomStyle AI")
st.write("Upload a photo of a room to detect its interior design style and get matching paint color suggestions.")

uploaded_file = st.file_uploader("Upload a room photo", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    img = Image.open(uploaded_file)
    st.image(img, caption="Uploaded Room", use_container_width=True)

    feat = extractor(img).reshape(1, -1)
    probs = model.predict_proba(feat)[0]
    idx = int(np.argmax(probs))
    label = le.inverse_transform([idx])[0]
    confidence = probs[idx] * 100

    name = "Bohemian" if label == "bohemain" else label.capitalize()
    st.subheader(f"Predicted Style: **{name}**")
    st.write(f"Confidence: {confidence:.1f}%")
    if confidence < 50:
        st.caption("Low confidence: colour-based features can't always separate similar styles.")

    st.subheader("Suggested Paint Colors")
    for color in color_recommendations.get(label, []):
        st.write(f"- {color}")

    with st.expander("See prediction breakdown across all styles"):
        for cls, p in zip(le.classes_, probs):
            st.write(f"{cls}: {p*100:.1f}%")
