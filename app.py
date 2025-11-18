import streamlit as st
import tensorflow as tf
import numpy as np
import pandas as pd
import os
import tempfile
from PIL import Image
from datetime import datetime
from zoneinfo import ZoneInfo

# ==== CONFIGURATION ====
TEST_IMAGE_DIR = "test_images"
LEADERBOARD_FILE = "leaderboard.csv"
CLASS_NAMES = ["A", "B", "C"]

# ==== LOAD ALL TEST IMAGES (do NOT resize here!) ====
@st.cache_data(show_spinner="Loading hidden test set…")
def load_raw_test_images():
    images, labels = [], []
    for idx, cls in enumerate(CLASS_NAMES):
        folder = os.path.join(TEST_IMAGE_DIR, cls)
        for fname in sorted(os.listdir(folder)):
            fpath = os.path.join(folder, fname)
            img = Image.open(fpath).convert("RGB")
            images.append(img)
            labels.append(idx)
    return images, np.array(labels)

def load_leaderboard():
    if os.path.exists(LEADERBOARD_FILE):
        return pd.read_csv(LEADERBOARD_FILE)
    else:
        return pd.DataFrame(columns=["Username", "Accuracy", "Timestamp"])

def save_leaderboard(df):
    df.to_csv(LEADERBOARD_FILE, index=False)

def evaluate_model(model, pil_images, y, input_size):
    resized_imgs = [np.array(img.resize(input_size)) / 255.0 for img in pil_images]
    x = np.stack(resized_imgs)
    preds = model.predict(x)
    y_pred = np.argmax(preds, axis=1)
    acc = (y_pred == y).mean()
    return acc

st.title("Sign Language Model Showdown!")
st.markdown("")

st.markdown(
    "Can your Keras model tell the difference between A, B, and C? ✊🖐️🤏 Time to put it to the test!"
)
st.markdown("")

st.markdown(
    "Upload your trained `.keras` model, and we’ll run it on our secret set of sign language photos. Once your model's evaluated, your score pops up on the leaderboard. Top the table, and those bragging rights are all yours! 🏆\n\n"
    "You can use almost any image size: 64×64, 128×128, 224×224, 256×256.Just make sure your model expects standard 3-channel (RGB) colour images."
)
st.markdown("")

col1, col2, col3 = st.columns([1, 12, 1])
with col2:
    st.markdown("👇 **Fill in your username, upload your model, and join the leaderboard. Good luck!** 👇")


username = st.text_input("Enter your username:")
uploaded_file = st.file_uploader(
    "Upload your Keras model (.keras only)", type=["keras"], accept_multiple_files=False
)
st.markdown("")

col1, col2, col3 = st.columns([1, 1, 1])
with col2:
    submit = st.button("Submit model for evaluation", type="primary")

leaderboard = load_leaderboard()
raw_images, y_test = load_raw_test_images()

if submit and uploaded_file and username.strip():
    with st.spinner("Evaluating your model..."):
        with tempfile.NamedTemporaryFile(suffix=".keras", delete=True) as tmpf:
            tmpf.write(uploaded_file.read())
            tmpf.flush()
            try:
                model = tf.keras.models.load_model(tmpf.name)
                input_shape = model.input_shape
                # Accept models with shape (None, H, W, 3)
                if len(input_shape) == 4 and input_shape[-1] == 3:
                    input_size = (input_shape[1], input_shape[2])  # (height, width)
                    if None in input_size:
                        st.error("Model input shape must have concrete dimensions, not None. Please specify fixed input size in your model.")
                        model = None
                    if model is not None:
                        try:
                            acc = evaluate_model(model, raw_images, y_test, input_size)
                            timestamp = datetime.now(ZoneInfo("Europe/Berlin")).strftime("%Y-%m-%d %H:%M:%S")
                            new_row = {
                                "Username": username,
                                "Accuracy": round(acc * 100, 2),
                                "Timestamp": timestamp,
                            }
                            leaderboard = pd.concat(
                                [leaderboard, pd.DataFrame([new_row])], ignore_index=True
                            )
                            save_leaderboard(leaderboard)
                            st.success(
                                f"🎉 All done! Your model scored {acc:.2%} on our test set. Your result has been added to the leaderboard."
                            )
                        except Exception as e:
                            st.error(f"Model could not be run on the test set: {e}")
                else:
                    st.error(
                        f"Model input shape {input_shape} is not supported. "
                        "Model must accept 3-channel (RGB) images."
                    )
            except Exception as e:
                st.error(
                    "Could not load your model file. "
                    "Please ensure it is a valid Keras `.keras` file. "
                    f"Error: {e}"
                )

elif submit and not uploaded_file:
    st.warning("Please upload your Keras `.keras` model file before submitting.")
elif submit and not username.strip():
    st.warning("Please enter a username before submitting.")

st.header("Leaderboard")
if leaderboard.empty:
    st.write("No submissions yet.")
else:
    leaderboard_display = leaderboard.copy()
    leaderboard_display["Accuracy"] = leaderboard_display["Accuracy"].astype(str) + " %"
    st.table(leaderboard_display.sort_values("Accuracy", ascending=False).reset_index(drop=True))

st.divider()

st.markdown(
    "You can submit as many models as you like. Each submission will appear as a new row in the leaderboard. Your uploaded model file is deleted after evaluation. "
)
