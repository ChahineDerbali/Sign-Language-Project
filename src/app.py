import streamlit as st
import cv2
import numpy as np
from tensorflow.keras.models import load_model

from predict import predict_letter

# Load the trained model
model = load_model(
    "C:/Users/CHPde/OneDrive/Bureau/Projects/Sign Language Project/ADS-Project/src/sign_language_model.h5"
)

# Set Streamlit's dark theme
st.markdown(
    """
    <style>
    body {
        background-color: #121212;
        color: white;
    }
    .streamlit-expanderHeader {
        color: white;
    }
    .stTextInput>div>input {
        background-color: #333;
        color: white;
    }
    .stButton>button {
        background-color: #6200EE;
        color: white;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Set Streamlit page title and description
st.title("Sign Language Recognition Demo")
st.write(
    "Use your webcam to sign letters, and the app will predict the corresponding letter."
)

# Initialize webcam
cap = cv2.VideoCapture(0)

# Create a placeholder for the webcam feed
image_placeholder = st.empty()

# Create an empty container to display the prediction
prediction_container = st.empty()

# Display instructions
st.markdown("""
    **Instructions:**
    1. Show your hand in front of the webcam.
    2. The model will predict the sign language letter based on your gesture.
    3. Press **'q'** to stop the webcam.
""")

# Start webcam loop
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Preprocess the frame: Flip, resize, normalize
    frame_resized = cv2.resize(frame, (64, 64))  # Resize
    frame_normalized = frame_resized / 255.0  # Normalize

    # Convert the normalized image back to uint8 (values in the range [0, 255])
    frame_uint8 = np.uint8(frame_normalized * 255)

    # Convert from BGR to RGB
    frame_rgb = cv2.cvtColor(frame_uint8, cv2.COLOR_BGR2RGB)

    # Add batch dimension
    frame_input = np.expand_dims(frame_rgb, axis=0)

    # Get prediction and confidence
    prediction = model.predict(frame_input)
    predicted_class = np.argmax(prediction)
    confidence = np.max(prediction)

    predicted_letter = chr(predicted_class + ord("A"))

    # Update the prediction in the container
    prediction_container.empty()  # Clear previous predictions
    prediction_container.write(
        f"Predicted: {predicted_letter} ({confidence * 100:.2f}%)"
    )

    # Update webcam preview on Streamlit app
    image_placeholder.image(frame, channels="BGR", use_container_width=True)

    # Exit on 'q' key press (no need for actual cv2.waitKey, since Streamlit runs asynchronously)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
