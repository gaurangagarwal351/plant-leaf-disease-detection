"""Run with: streamlit run app.py"""

from pathlib import Path

import streamlit as st
from PIL import Image

from leaf_disease.predict import predict_image
from leaf_disease.runtime import load_model, select_device


st.set_page_config(page_title="Plant Leaf Health", page_icon="🌿")
st.title("🌿 Plant Leaf Health")
st.write("Upload a clear photo of one leaf to estimate whether it looks healthy or diseased.")


@st.cache_resource
def get_model(path: str):
    device = select_device("auto")
    model, class_to_idx, image_size = load_model(Path(path), device)
    return model, class_to_idx, image_size, device


checkpoint = Path("outputs/best_model.pt")
if not checkpoint.is_file():
    st.info("Train a model first. See README.md for setup and training commands.")
    st.stop()

uploaded = st.file_uploader("Leaf image", type=("jpg", "jpeg", "png", "webp"))
if uploaded is not None:
    try:
        with Image.open(uploaded) as image:
            st.image(image, caption="Uploaded leaf", width=400)
            model, class_to_idx, image_size, device = get_model(str(checkpoint))
            result = predict_image(model, image, class_to_idx, image_size, device)
        st.subheader(result["prediction"].capitalize())
        st.write(f"Confidence: {result['probabilities'][result['prediction']]:.1%}")
        st.write("Class scores:")
        st.json({key: round(value, 4) for key, value in result["probabilities"].items()})
        st.caption("A model estimate is not a plant diagnosis. Check uncertain results with a local expert.")
    except (OSError, ValueError) as exc:
        st.error(f"Could not process this image: {exc}")
