"""
SWYNEX Task 2 - Model Integration Prototype
Graphic Design Style Classifier (Streamlit web app)

Reuses the trained Keras model from Task 1:
  https://github.com/maniram02112006/SWYNEX-AI-Graphic-Design-Classifier

Facts taken from the Task 1 source code and the saved model file:
  - Input size : 160 x 160 pixels, 3 colour channels (RGB)
  - Pixel range: raw 0-255 values. MobileNetV2 preprocessing (x / 127.5 - 1)
                 is saved INSIDE the model, so the app must NOT rescale again.
  - Output     : 5 softmax probabilities
  - Class order: the order of the `categories` list in Task 1
                 (this is NOT alphabetical order)
"""

from pathlib import Path

import numpy as np
import streamlit as st
from PIL import Image, UnidentifiedImageError

# ----------------------------------------------------------------------
# Settings (all taken from the Task 1 training script)
# ----------------------------------------------------------------------
MODEL_PATH = Path(__file__).parent / "model" / "graphic_design_classifier.keras"
EXAMPLES_DIR = Path(__file__).parent / "examples"
IMAGE_SIZE = (160, 160)
CLASS_NAMES = ["cinematic", "minimalist", "typography", "sports", "vintage"]

ALLOWED_TYPES = ["jpg", "jpeg", "png"]
MAX_FILE_MB = 10
LOW_CONFIDENCE = 0.50  # simple UI warning level (not a scientific threshold)


# ----------------------------------------------------------------------
# Model loading (cached so it loads only once)
# ----------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading the trained model...")
def load_classifier():
    """Load the Task 1 Keras model. Returns (model, error_message)."""
    if not MODEL_PATH.exists():
        return None, (
            f"Model file not found at: {MODEL_PATH}\n\n"
            "Download 'graphic_design_classifier.keras' from the Task 1 "
            "repository and place it inside the 'model' folder."
        )
    try:
        import keras  # imported here so a missing install gives a clear message

        model = keras.models.load_model(MODEL_PATH, compile=False)
    except Exception as error:  # noqa: BLE001 - show any loading problem
        return None, f"The model file could not be loaded: {error}"

    # Safety check: the model must match the class list in this app.
    output_units = model.output_shape[-1]
    if output_units != len(CLASS_NAMES):
        return None, (
            f"The model has {output_units} outputs but this app defines "
            f"{len(CLASS_NAMES)} classes. Check CLASS_NAMES in app.py."
        )
    return model, None


# ----------------------------------------------------------------------
# Image handling
# ----------------------------------------------------------------------
def open_image(uploaded_file):
    """Validate and open the uploaded file. Returns (image, error_message)."""
    size_mb = uploaded_file.size / (1024 * 1024)
    if size_mb > MAX_FILE_MB:
        return None, f"The file is {size_mb:.1f} MB. Please upload an image under {MAX_FILE_MB} MB."
    try:
        # verify() checks the file is a real image; it must be re-opened after.
        Image.open(uploaded_file).verify()
        uploaded_file.seek(0)
        image = Image.open(uploaded_file)
        image.load()
    except (UnidentifiedImageError, OSError, SyntaxError):
        return None, "This file is not a valid or readable JPG/PNG image."
    if image.width < 32 or image.height < 32:
        return None, "The image is too small (minimum 32 x 32 pixels)."
    return image, None


def preprocess(image):
    """
    Match the Task 1 training pipeline:
      keras.utils.load_img(path, target_size=(160, 160))   -> RGB, nearest resize
      keras.utils.img_to_array(...)                        -> float32, 0-255
    The model itself applies MobileNetV2 preprocessing, so no /255 here.
    """
    image = image.convert("RGB")
    image = image.resize(IMAGE_SIZE, Image.NEAREST)
    array = np.asarray(image, dtype="float32")
    return np.expand_dims(array, axis=0)  # shape (1, 160, 160, 3)


def predict(model, image):
    batch = preprocess(image)
    probabilities = model.predict(batch, verbose=0)[0]
    return probabilities


# ----------------------------------------------------------------------
# User interface
# ----------------------------------------------------------------------
st.set_page_config(page_title="Graphic Design Style Classifier", page_icon="🎨", layout="centered")

st.title("🎨 Graphic Design Style Classifier")
st.caption("SWYNEX Technologies Internship - Task 2: Model Integration Prototype")
st.write(
    "Upload a poster or graphic design image (JPG or PNG). A MobileNetV2-based "
    "Keras model trained in Task 1 predicts one of five design styles: "
    + ", ".join(CLASS_NAMES) + "."
)

model, load_error = load_classifier()
if load_error:
    st.error(load_error)
    st.stop()

input_mode = st.radio(
    "Choose image source:",
    ["Upload your own image", "Try an example from dataset"],
    horizontal=True,
)

image = None
image_caption = ""

if input_mode == "Upload your own image":
    uploaded = st.file_uploader("Choose an image (JPG or PNG)", type=ALLOWED_TYPES)
    if uploaded is not None:
        image, image_error = open_image(uploaded)
        if image_error:
            st.error(image_error)
            image = None
        else:
            image_caption = uploaded.name
    else:
        st.info("Upload a JPG or PNG image to see a prediction.")
else:
    sample_files = sorted(EXAMPLES_DIR.glob("*.jpg")) if EXAMPLES_DIR.exists() else []
    if not sample_files:
        st.warning("No sample images found in the 'examples' folder.")
    else:
        sample_names = [f.name for f in sample_files]
        chosen = st.selectbox("Select a sample image:", sample_names)
        chosen_path = EXAMPLES_DIR / chosen
        try:
            image = Image.open(chosen_path)
            image.load()
            image_caption = f"Sample: {chosen}"
        except Exception as err:  # noqa: BLE001
            st.error(f"Could not load sample image: {err}")
            image = None

if image is not None:
    left, right = st.columns(2)
    with left:
        st.image(image, caption=image_caption, use_container_width=True)
    with right:
        try:
            with st.spinner("Classifying..."):
                probabilities = predict(model, image)
        except Exception as error:  # noqa: BLE001
            st.error(f"Prediction failed: {error}")
        else:
            best = int(np.argmax(probabilities))
            confidence = float(probabilities[best])
            st.subheader(f"Predicted style: {CLASS_NAMES[best].title()}")
            st.metric("Model confidence", f"{confidence * 100:.1f}%")
            if confidence < LOW_CONFIDENCE:
                st.warning(
                    "Low confidence: the design may mix several styles, "
                    "so treat this prediction with caution."
                )

    st.markdown("#### Probability for each style")
    for name, prob in sorted(zip(CLASS_NAMES, probabilities), key=lambda p: -p[1]):
        st.write(f"**{name.title()}** - {prob * 100:.1f}%")
        st.progress(float(min(max(prob, 0.0), 1.0)))

st.divider()
with st.expander("About this prototype and its limits"):
    st.markdown(
        """
- The model was trained in Task 1 on a **very small sample set (20 AI-generated
  illustrative images, 4 per style)**. No reliable accuracy has been measured.
- The confidence value is the model's softmax output. It is **not** a calibrated
  probability and can be high even when the prediction is wrong.
- Images that do not look like posters, or that mix several styles, may be
  classified poorly. The model always picks one of the five styles.
- This is a learning prototype, not a production tool.
"""
    )
