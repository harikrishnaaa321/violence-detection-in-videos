# import gradio as gr
# import cv2
# import numpy as np
# import tensorflow as tf
# from tensorflow.keras.models import load_model
# import time
#
# # Load the trained model
# model = load_model("model/mob_bilstm_model.keras")
#
# # Constants
# IMAGE_HEIGHT, IMAGE_WIDTH = 64, 64
# SEQUENCE_LENGTH = 16
# CLASSES_LIST = ["NonViolence", "Violence"]
#
#
# def preprocess_video(video_path):
#     """Extracts frames from a video, resizes them, and normalizes for model input."""
#     frames_list = []
#     original_frames = []
#     cap = cv2.VideoCapture(video_path)
#     total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
#
#     if total_frames < SEQUENCE_LENGTH:
#         return None, None  # Not enough frames
#
#     skip_frames = max(total_frames // SEQUENCE_LENGTH, 1)
#
#     for i in range(SEQUENCE_LENGTH):
#         cap.set(cv2.CAP_PROP_POS_FRAMES, i * skip_frames)
#         success, frame = cap.read()
#         if not success:
#             break
#         frame_resized = cv2.resize(frame, (IMAGE_WIDTH, IMAGE_HEIGHT))
#         frame_normalized = frame_resized / 255.0  # Normalize pixel values
#         frames_list.append(frame_normalized)
#         original_frames.append(cv2.resize(frame, (120, 120)))  # Display size
#
#     cap.release()
#
#     if len(frames_list) == SEQUENCE_LENGTH:
#         return np.expand_dims(np.array(frames_list), axis=0), original_frames
#     else:
#         return None, None
#
#
# def predict_violence(video_path):
#     """Runs the model on a video file and returns predictions with progress updates."""
#     if not video_path:
#         yield "⚠️ Please upload a video!", None
#
#     start_time = time.time()
#
#     yield "⏳ (1/7) Preprocessing video...", None
#     time.sleep(0.5)
#
#     frames, original_frames = preprocess_video(video_path)
#     if frames is None:
#         yield "❌ (2/7) Error: Not enough frames in the video!", None
#         return
#
#     yield "⏳ (3/7) Loading AI model...", None
#     time.sleep(0.5)
#
#     yield "🔍 (4/7) Analyzing video... Please wait ⏳", None
#     time.sleep(1)
#
#     predictions = model.predict(frames)
#
#     yield "⏳ (5/7) Processing AI predictions...", None
#     time.sleep(0.5)
#
#     predicted_label = CLASSES_LIST[np.argmax(predictions)]
#     confidence = np.max(predictions) * 100
#
#     yield "⏳ (6/7) Finalizing results... Almost done!", None
#     time.sleep(0.5)
#
#     end_time = time.time()
#     processing_time = end_time - start_time
#
#     output_text = f"✅ Prediction: **{predicted_label}** ({confidence:.2f}%)\n⏳ Processing Time: {processing_time:.2f} seconds"
#
#     # Overlay label text on frames (but do not save them)
#     for i, frame in enumerate(original_frames[:4]):  # Display 4 sample frames
#         label = "Violence" if predicted_label == "Violence" else "NonViolence"
#         original_frames[i] = cv2.putText(frame, label, (5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
#
#     yield "🎉 (7/7) Analysis Complete!", None
#     time.sleep(0.5)
#
#     yield output_text, original_frames
#
#
# # Gradio UI
# with gr.Blocks() as demo:
#     gr.HTML("""
#     <div style="text-align: center; padding: 20px; background-color: #FF5733; color: white; border-radius: 10px;">
#         <h1 style="margin: 0;">🚀 AI-Powered Violence Detection System 🛡️</h1>
#         <p style="margin: 0; font-size: 18px;">Upload a video to analyze and detect violence using deep learning.</p>
#     </div>
#     """)
#
#     with gr.Row():
#         with gr.Column(scale=1.2, min_width=320):
#             gr.Markdown("## 📌 **Instructions**")
#             gr.Markdown("""
#             - Upload a **short video** 📹
#             - AI will **analyze it** for violence detection 🔍
#             - Get a **detailed prediction** with confidence score ✅
#             - Sample frames **will be displayed** below 📸
#             """)
#             gr.Markdown("---")
#             gr.Markdown("## 🤖 **Model Info**")
#             gr.Markdown("""
#             - **Architecture:** MobileNet + BiLSTM
#             - **Frames Used:** 16 key frames
#             - **Categories:** Violence / NonViolence
#             """)
#
#         with gr.Column(scale=3):
#             gr.Markdown("## 🛡️ **Upload Video for Analysis**")
#
#             with gr.Row():
#                 video_input = gr.Video(label="🎥 Upload Video", interactive=True)
#                 video_output = gr.Textbox(label="Prediction", interactive=False, show_label=False)
#
#             with gr.Row():
#                 analyze_button = gr.Button("🔍 Reanalyze Video", variant="primary")
#                 clear_button = gr.Button("🗑️ Clear Output", variant="secondary")
#
#             frames_output = gr.Gallery(label="📸 Sample Frames", columns=4, height=150)
#
#             video_input.change(predict_violence, inputs=video_input, outputs=[video_output, frames_output])
#             analyze_button.click(predict_violence, inputs=video_input, outputs=[video_output, frames_output])
#             clear_button.click(lambda: ("", None), outputs=[video_output, frames_output])
#
# # Launch the app
# if __name__ == "__main__":
#     demo.launch()


import gradio as gr
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import time
from huggingface_hub import hf_hub_download

# Download the model from Hugging Face Hub
MODEL_PATH = hf_hub_download(repo_id="harikrishnaaa321/my-violence-detection-model", filename="mob_bilstm_model.keras")
model = load_model(MODEL_PATH)

# Constants
IMAGE_HEIGHT, IMAGE_WIDTH = 64, 64
SEQUENCE_LENGTH = 16
CLASSES_LIST = ["NonViolence", "Violence"]

def preprocess_video(video_path):
    """Extracts frames from a video, resizes them, and normalizes for model input."""
    frames_list = []
    original_frames = []
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if total_frames < SEQUENCE_LENGTH:
        return None, None  # Not enough frames

    skip_frames = max(total_frames // SEQUENCE_LENGTH, 1)

    for i in range(SEQUENCE_LENGTH):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i * skip_frames)
        success, frame = cap.read()
        if not success:
            break
        frame_resized = cv2.resize(frame, (IMAGE_WIDTH, IMAGE_HEIGHT))
        frame_normalized = frame_resized / 255.0  # Normalize pixel values
        frames_list.append(frame_normalized)
        original_frames.append(cv2.resize(frame, (120, 120)))  # Display size

    cap.release()

    if len(frames_list) == SEQUENCE_LENGTH:
        return np.expand_dims(np.array(frames_list), axis=0), original_frames
    else:
        return None, None


def predict_violence(video_path):
    """Runs the model on a video file and returns predictions with progress updates."""
    if not video_path:
        yield "⚠️ Please upload a video!", None

    start_time = time.time()

    yield "⏳ (1/7) Preprocessing video...", None
    time.sleep(0.5)

    frames, original_frames = preprocess_video(video_path)
    if frames is None:
        yield "❌ (2/7) Error: Not enough frames in the video!", None
        return

    yield "⏳ (3/7) Loading AI model...", None
    time.sleep(0.5)

    yield "🔍 (4/7) Analyzing video... Please wait ⏳", None
    time.sleep(1)

    predictions = model.predict(frames)

    yield "⏳ (5/7) Processing AI predictions...", None
    time.sleep(0.5)

    predicted_label = CLASSES_LIST[np.argmax(predictions)]
    confidence = np.max(predictions) * 100

    yield "⏳ (6/7) Finalizing results... Almost done!", None
    time.sleep(0.5)

    end_time = time.time()
    processing_time = end_time - start_time

    output_text = f"✅ Prediction: **{predicted_label}** ({confidence:.2f}%)\n⏳ Processing Time: {processing_time:.2f} seconds"

    # Overlay label text on frames (but do not save them)
    for i, frame in enumerate(original_frames[:4]):  # Display 4 sample frames
        label = "Violence" if predicted_label == "Violence" else "NonViolence"
        original_frames[i] = cv2.putText(frame, label, (5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

    yield "🎉 (7/7) Analysis Complete!", None
    time.sleep(0.5)

    yield output_text, original_frames


# Gradio UI
with gr.Blocks() as demo:
    gr.HTML("""
    <div style="text-align: center; padding: 20px; background-color: #FF5733; color: white; border-radius: 10px;">
        <h1 style="margin: 0;">🚀 AI-Powered Violence Detection System 🛡️</h1>
        <p style="margin: 0; font-size: 18px;">Upload a video to analyze and detect violence using deep learning.</p>
    </div>
    """)

    with gr.Row():
        with gr.Column(scale=1.2, min_width=320):
            gr.Markdown("## 📌 **Instructions**")
            gr.Markdown("""
            - Upload a **short video** 📹
            - AI will **analyze it** for violence detection 🔍
            - Get a **detailed prediction** with confidence score ✅
            - Sample frames **will be displayed** below 📸
            """)
            gr.Markdown("---")
            gr.Markdown("## 🤖 **Model Info**")
            gr.Markdown("""
            - **Architecture:** MobileNet + BiLSTM
            - **Frames Used:** 16 key frames
            - **Categories:** Violence / NonViolence
            """)

        with gr.Column(scale=3):
            gr.Markdown("## 🛡️ **Upload Video for Analysis**")

            with gr.Row():
                video_input = gr.Video(label="🎥 Upload Video", interactive=True)
                video_output = gr.Textbox(label="Prediction", interactive=False, show_label=False)

            with gr.Row():
                analyze_button = gr.Button("🔍 Reanalyze Video", variant="primary")
                clear_button = gr.Button("🗑️ Clear Output", variant="secondary")

            frames_output = gr.Gallery(label="📸 Sample Frames", columns=4, height=150)

            video_input.change(predict_violence, inputs=video_input, outputs=[video_output, frames_output])
            analyze_button.click(predict_violence, inputs=video_input, outputs=[video_output, frames_output])
            clear_button.click(lambda: ("", None), outputs=[video_output, frames_output])

# Launch the app
if __name__ == "__main__":
    demo.launch()
