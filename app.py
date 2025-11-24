import streamlit as st
import tensorflow as tf
import numpy as np
import pandas as pd
import os
import gc
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

def evaluate_model_batched(model, pil_images, y_true, input_size, batch_size=32):
    """
    Evaluates model accuracy using batch processing to prevent Memory Spikes.
    """
    total_correct = 0
    total_images = len(pil_images)
    
    # Create a progress bar since batching might take a moment
    progress_bar = st.progress(0)
    
    # Process in chunks
    for i in range(0, total_images, batch_size):
        batch_indices = slice(i, i + batch_size)
        batch_imgs = pil_images[batch_indices]
        batch_y = y_true[batch_indices]
        
        # 1. Resize only this batch
        # Note: We normalize to [0,1] immediately if your models expect float inputs. 
        # If models expect 0-255, remove the / 255.0. 
        # Most standard Keras models expect floats or have a Rescaling layer.
        # For safety, we keep it as raw array matching your original logic.
        processed_batch = [
            np.array(img.resize(input_size)) 
            for img in batch_imgs
        ]
        
        # 2. Stack only this batch
        x_batch = np.stack(processed_batch)
        
        # 3. Predict only this batch
        # verbose=0 prevents printing to stdout which can slow down Streamlit
        preds = model.predict(x_batch, verbose=0)
        
        # 4. Calculate accuracy for this batch
        y_pred = np.argmax(preds, axis=1)
        total_correct += np.sum(y_pred == batch_y)
        
        # Update progress
        progress_bar.progress(min((i + batch_size) / total_images, 1.0))
        
        # 5. Manual cleanup for this loop iteration
        del x_batch, processed_batch, preds
    
    progress_bar.empty() # Remove bar when done
    
    return total_correct / total_images

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
            
            model = None # Initialize variable
            
            try:
                # Load the model
                model = tf.keras.models.load_model(tmpf.name)
                input_shape = model.input_shape
                
                # Validation logic
                if len(input_shape) == 4 and input_shape[-1] == 3:
                    input_size = (input_shape[1], input_shape[2])
                    
                    if None in input_size:
                         st.error("Model input shape must have concrete dimensions...")
                    else:
                        try:
                            # Use the NEW batched evaluator
                            acc = evaluate_model_batched(model, raw_images, y_test, input_size)
                            
                            # ... [Save Leaderboard Logic remains the same] ...
                            
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
                            
                            st.success(f"🎉 All done! Score: {acc:.2%}")
                            
                        except Exception as e:
                            st.error(f"Model run error: {e}")
                else:
                    st.error(f"Invalid Input Shape: {input_shape}")
                    
            except Exception as e:
                st.error(f"Load Error: {e}")
                
            finally:
                # ==== CRITICAL MEMORY CLEANUP ====
                if model:
                    del model
                
                # 1. Clear TensorFlow Session/Graph
                tf.keras.backend.clear_session()
                
                # 2. Force Python Garbage Collection
                gc.collect()
                

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
