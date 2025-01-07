import streamlit as st
import joblib
import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from pathlib import Path
from src.train import SELECTED_COLUMNS
from src.utils import get_rect_points

# Load the classifier model
@st.cache_resource
def load_clf_model(model_path: str):
    return joblib.load(model_path)

# Load the Mediapipe model
@st.cache_resource
def load_mediapipe_model(model_path: str):
    base_options = python.BaseOptions(model_asset_path=model_path)
    options = vision.FaceLandmarkerOptions(
        base_options=base_options,
        output_face_blendshapes=True,
        output_facial_transformation_matrixes=True,
        num_faces=1,
    )
    return vision.FaceLandmarker.create_from_options(options)

# Function to process and predict
def process_frame(frame, clf, landmarker):
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_frame = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
    detection_result = landmarker.detect(mp_frame)

    if len(detection_result.face_blendshapes) > 0:
        face_blendshapes = detection_result.face_blendshapes[0]
        scores_dict = {blendshape.category_name: blendshape.score for blendshape in face_blendshapes}
        ordered_scores = [scores_dict.get(column, 0.0) for column in SELECTED_COLUMNS]
        y_pred = clf.predict([ordered_scores])
        confidence = max(clf.predict_proba([ordered_scores])[0])  # Get prediction confidence

        height, width, _ = frame.shape
        x_min_px, x_max_px, y_min_px, y_max_px = get_rect_points(detection_result.face_landmarks[0], height, width)

        return y_pred[0], confidence, (x_min_px, x_max_px, y_min_px, y_max_px)
    return None, None, None

# Streamlit App
def main():
    st.set_page_config(page_title="Drowsiness Detection App", layout="wide")

    # Title and description
    st.title("Drowsiness Detection System")
    st.markdown("""
    This app uses your webcam to detect drowsiness in real-time.  
    The predictions are powered by a Random Forest classifier and Mediapipe's Face Landmark model.
    """)

    # Model paths
    root_path = str(Path(__file__).parent)
    model_path = root_path + "/models/random_forest_model_1.pkl"
    mediapipe_model_path = root_path + "/face_landmarker_v2_with_blendshapes.task"

    # Load models
    clf = load_clf_model(model_path)
    landmarker = load_mediapipe_model(mediapipe_model_path)

    # Layout: Webcam on left, predictions on right
    col1, col2 = st.columns([3, 2])

    with col1:
        st.header("Webcam Feed")
        run = st.checkbox("Start Webcam")
        FRAME_WINDOW = st.image([])

    with col2:
        st.header("Prediction Details")
        prediction_placeholder = st.empty()
        confidence_placeholder = st.empty()

    cap = cv2.VideoCapture(0)  # Open webcam

    while run:
        success, frame = cap.read()
        if not success:
            st.warning("Webcam not detected!")
            break

        prediction, confidence, bbox = process_frame(frame, clf, landmarker)

        if prediction:
            if prediction == 'normal':
                font_color = (0, 255, 0)
            elif prediction == 'yawning':
                font_color = (0, 0, 255)

            x_min_px, x_max_px, y_min_px, y_max_px = bbox
            cv2.putText(frame, prediction, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, font_color, 2, cv2.LINE_AA)
            cv2.rectangle(frame, (x_min_px, y_min_px), (x_max_px, y_max_px), font_color, 2)

            # Update prediction details
            prediction_placeholder.markdown(f"### Predicted State: **{prediction.capitalize()}**")
            confidence_placeholder.markdown(f"### Confidence: **{confidence * 100:.2f}%**")

        else:
            prediction_placeholder.markdown("### Predicted State: **No Face Detected**")
            confidence_placeholder.markdown("### Confidence: N/A")

        # Display in Streamlit
        FRAME_WINDOW.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

