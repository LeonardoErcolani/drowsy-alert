import streamlit as st
import cv2
import mediapipe as mp
import pandas as pd
from collections import deque
import numpy as np
import joblib
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from src.train import SELECTED_COLUMNS
from src.utils import get_rect_points
import time

# Load the classifier model

def load_clf_model(model_path: str):
    return joblib.load(model_path)

# Load the Mediapipe model

def load_mediapipe_model(model_path: str):
    base_options = python.BaseOptions(model_asset_path=model_path)
    options = vision.FaceLandmarkerOptions(
        base_options=base_options,
        output_face_blendshapes=True,
        output_facial_transformation_matrixes=True,
        num_faces=1,
    )
    return vision.FaceLandmarker.create_from_options(options)

# Function to process frame and predict yawning
def process_frame_with_prediction(frame, clf, landmarker):
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_frame = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
    detection_result = landmarker.detect(mp_frame)

    if len(detection_result.face_blendshapes) > 0:
        face_blendshapes = detection_result.face_blendshapes[0]
        scores_dict = {blendshape.category_name: blendshape.score for blendshape in face_blendshapes}
        ordered_scores = [scores_dict.get(column, 0.0) for column in SELECTED_COLUMNS]
        y_pred = clf.predict([ordered_scores])[0]
        confidence = max(clf.predict_proba([ordered_scores])[0])  # Confidence of prediction

        height, width, _ = frame.shape
        x_min_px, x_max_px, y_min_px, y_max_px = get_rect_points(detection_result.face_landmarks[0], height, width)

        return y_pred, confidence, (x_min_px, x_max_px, y_min_px, y_max_px)
    return None, None, None

# Initialize MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
)

# Streamlit Layout
st.set_page_config(layout="wide")

st.title("Drowsy Alert System with Yawning Prediction")

# Create columns for the layout
col1, col2 = st.columns([4, 4])  # Left column (for webcam) is wider

# Webcam Feed on Left (col1)
with col1:
    st.markdown(
    """
    ### Real-Time Monitoring
    This application detects drowsiness using face landmarks and predicts yawning in real-time using a machine learning model.
    - **Mouth Openness Ratio**
    - **Eye Aspect Ratio (EAR)**
    - **Yawning Prediction**
    """
    )
    # Start Webcam and Detection
    run = st.toggle("Activate Detection")
    FRAME_WINDOW = st.image([])

    st.markdown("**Yawning Prediction**")
    prediction_placeholder = st.empty()
    confidence_placeholder = st.empty()

# Graphs and Predictions on Right (col2 and col3)
with col2:
    st.subheader("Live Metrics")
    st.markdown("1. **Mouth Openness Ratio**")
    mouth_placeholder = st.empty()
    st.markdown("2. **Left Eye Aspect Ratio (EAR)**")
    left_eye_placeholder = st.empty()
    st.markdown("2. **Right Eye Aspect Ratio (EAR)**")
    right_eye_placeholder= st.empty()



# Load models
model_path = "./models/random_forest_model_1.pkl"
mediapipe_model_path = "./face_landmarker_v2_with_blendshapes.task"
clf = load_clf_model(model_path)
landmarker = load_mediapipe_model(mediapipe_model_path)

# Function to calculate distances between two landmarks
def calculate_distance(landmark1, landmark2, frame_shape):
    height, width, _ = frame_shape
    x1, y1 = int(landmark1.x * width), int(landmark1.y * height)
    x2, y2 = int(landmark2.x * width), int(landmark2.y * height)
    return np.sqrt((x2 - x1)**2 + (y2 - y1)**2)

# Function to calculate EAR (Eye Aspect Ratio)
def calculate_ear(eye_landmarks, frame_shape):
    vertical1 = calculate_distance(eye_landmarks[1], eye_landmarks[5], frame_shape)
    vertical2 = calculate_distance(eye_landmarks[2], eye_landmarks[4], frame_shape)
    horizontal = calculate_distance(eye_landmarks[0], eye_landmarks[3], frame_shape)
    return (vertical1 + vertical2) / (2.0 * horizontal)

# Function to calculate mouth openness ratio
def calculate_mouth_openness(landmarks, frame_shape):
    vertical = calculate_distance(landmarks[13], landmarks[14], frame_shape)
    horizontal = calculate_distance(landmarks[78], landmarks[308], frame_shape)
    return vertical / horizontal

# Deques for graph data
time_data = deque(maxlen=50)
mouth_data = deque(maxlen=50)
left_eye_data = deque(maxlen=50)
right_eye_data = deque(maxlen=50)

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    st.error("Unable to access webcam. Please check your camera settings.")
    st.stop()

# Variables to track eye closure duration
eye_closed_time = 0
eye_threshold = 0.2  # Threshold for determining if the eyes are closed
warning_duration = 2  # Time in seconds for both eyes to be closed
warning_shown = False

# Variables to track yawning duration
yawning_start_time = None
yawning_duration_threshold = 2  # seconds
warning_shown_yawn = False

while run:
    ret, frame = cap.read()
    if not ret:
        st.warning("Failed to capture frame. Please check your webcam.")
        break

    # Process frame for yawning prediction
    prediction, confidence, bbox = process_frame_with_prediction(frame, clf, landmarker)

    # Draw landmarks and metrics
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_frame)

    if results.multi_face_landmarks:
        for landmarks in results.multi_face_landmarks:
            mp_drawing.draw_landmarks(
                image=frame,
                landmark_list=landmarks,
                connections=mp_face_mesh.FACEMESH_CONTOURS,
                landmark_drawing_spec=None,
            )
            # Calculate metrics
            mouth_openness = calculate_mouth_openness(landmarks.landmark, frame.shape)
            left_eye_indices = [362, 385, 387, 263, 373, 380]
            right_eye_indices = [33, 160, 158, 133, 153, 144]
            left_eye_landmarks = [landmarks.landmark[i] for i in left_eye_indices]
            right_eye_landmarks = [landmarks.landmark[i] for i in right_eye_indices]
            left_ear = calculate_ear(left_eye_landmarks, frame.shape)
            right_ear = calculate_ear(right_eye_landmarks, frame.shape)

            # Update data for graphs
            time_data.append(len(time_data))
            mouth_data.append(mouth_openness)
            left_eye_data.append(left_ear)
            right_eye_data.append(right_ear)

            # Draw a bounding box
            #x_min = int(min([lm.x for lm in landmarks.landmark]) * frame.shape[1])
            #y_min = int(min([lm.y for lm in landmarks.landmark]) * frame.shape[0])
            #x_max = int(max([lm.x for lm in landmarks.landmark]) * frame.shape[1])
            #y_max = int(max([lm.y for lm in landmarks.landmark]) * frame.shape[0])
            #cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (255, 255, 0), 2)

            

            # Update graphs
            mouth_chart = pd.DataFrame({"Time": list(time_data), "Mouth Openness": list(mouth_data)})
            left_eye_chart = pd.DataFrame({"Time": list(time_data), "Left EAR": list(left_eye_data)})
            right_eye_chart = pd.DataFrame({"Time": list(time_data), "Right EAR": list(right_eye_data)})
            
            mouth_placeholder.line_chart(mouth_chart["Mouth Openness"], x_label= "Time", y_label="Mouth Openness", width= 500, height=300, use_container_width=False)
            left_eye_placeholder.line_chart(left_eye_chart["Left EAR"], x_label= "Time", y_label="Left Eye EAR",  width= 500 ,height=300,use_container_width=False)
            right_eye_placeholder.line_chart(right_eye_chart["Right EAR"], x_label= "Time", y_label="Right Eye EAR", width= 500 ,height=300,use_container_width=False)

    with col1:
        FRAME_WINDOW.image(frame, channels="BGR")

        # Display yawning prediction
        if prediction == "yawning":
            if yawning_start_time is None:
                yawning_start_time = time.time()
            elif time.time() - yawning_start_time > yawning_duration_threshold and not warning_shown_yawn:
                st.warning("Attention! Yawning for more than 2 seconds!")
                warning_shown_yawn = True
        else:
            yawning_start_time = None
            warning_shown_yawn = False  # Reset timer if no yawning is detected

        # Check if both eyes are closed
        if left_ear < eye_threshold and right_ear < eye_threshold:
            if eye_closed_time == 0:
                    eye_closed_time = time.time()
            elif time.time() - eye_closed_time > warning_duration and not warning_shown:
                    st.warning("Both eyes are closed for more than 2 seconds! Please wake up!")
                    warning_shown = True
            else:
                eye_closed_time = 0  # Reset timer if eyes are not closed
                warning_shown = False

    if prediction:
        color = (0, 255, 0) if prediction == "normal" else (0, 0, 255)
        cv2.putText(frame, f"{prediction.capitalize()} ({confidence:.2f})", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        if bbox:
            x_min, x_max, y_min, y_max = bbox
            cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), color, 2)
        prediction_placeholder.markdown(f"### Predicted State: **{prediction.capitalize()}**")
        confidence_placeholder.markdown(f"### Confidence: ***{confidence * 100:.2f}%***")

    # Show frame
    FRAME_WINDOW.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

cap.release()
cv2.destroyAllWindows()
