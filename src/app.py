import streamlit as st
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image


@st.cache_resource
def get_model():
    return load_model(
        "C:/Users/CHPde/OneDrive/Bureau/Projects/Sign Language Project/ADS-Project/models/sign_language_model_cropped100.h5"
    )


model = get_model()

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

st.title("Sign Language Recognition Demo")
st.write("Choose an option to test the model:")


option = st.radio("Choose Input Method", ("Webcam", "Upload Image"))


def predict_image(image):
    # Resize to the expected model input size (64x64)
    image_resized = cv2.resize(image, (100, 100))
    image_normalized = image_resized / 255.0  # Normalize pixel values

    # Convert the normalized image back to uint8 (values in the range [0, 255])
    image_uint8 = np.uint8(image_normalized * 255)

    image_rgb = cv2.cvtColor(image_uint8, cv2.COLOR_BGR2RGB)  # Convert to RGB

    # Add batch dimension to the image
    image_input = np.expand_dims(image_rgb, axis=0)

    # Get prediction and confidence for the uploaded image
    prediction = model.predict(image_input)
    predicted_class = np.argmax(prediction)
    confidence = np.max(prediction)

    predicted_letter = chr(predicted_class + ord("A"))

    # Display predicted letter and confidence
    st.write(f"Predicted: {predicted_letter} ({confidence * 100:.2f}%)")


# Webcam Option
if option == "Webcam":
    st.header("Webcam Feed - Show your hand in the center of the screen")

    # Initialize webcam
    cap = cv2.VideoCapture(0)

    # Create a placeholder for the webcam feed
    image_placeholder = st.empty()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Get the height and width of the frame
        height, width, _ = frame.shape

        # Calculate the center and the size of the square
        min_side = min(height, width)
        center_x, center_y = width // 2, height // 2

        # Crop the frame to a square (center the crop)
        cropped_frame = frame[
            center_y - min_side // 2 : center_y + min_side // 2,
            center_x - min_side // 2 : center_x + min_side // 2,
        ]

        # Preprocess the cropped square frame (for hand detection)
        cropped_resized = cv2.resize(
            cropped_frame, (100, 100)
        )  # Resize to model input size (64x64)
        cropped_normalized = cropped_resized / 255.0  # Normalize pixel values

        # Convert the normalized image back to uint8 (values in the range [0, 255])
        cropped_uint8 = np.uint8(cropped_normalized * 255)

        # Convert from BGR to RGB (now that the image is uint8)
        cropped_rgb = cv2.cvtColor(cropped_uint8, cv2.COLOR_BGR2RGB)

        # Add batch dimension
        cropped_input = np.expand_dims(cropped_rgb, axis=0)

        # Get prediction and confidence for the cropped square frame
        prediction = model.predict(cropped_input)
        predicted_class = np.argmax(prediction)
        confidence = np.max(prediction)

        predicted_letter = chr(predicted_class + ord("A"))

        # Display predicted letter and confidence on the webcam feed
        cv2.putText(
            frame,
            f"Predicted: {predicted_letter} ({confidence * 100:.2f}%)",  # Show confidence as percentage
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 0, 0),
            2,
        )

        # Show the live feed with the cropped area
        image_placeholder.image(frame, channels="BGR", use_container_width=True)

        # Exit on 'q' key press (use Streamlit's callback)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()

# Image Upload Option
elif option == "Upload Image":
    st.header("Upload an Image to Test Prediction")

    # File uploader widget
    uploaded_file = st.file_uploader("Choose an image", type=["jpg", "png", "jpeg"])

    if uploaded_file is not None:
        # Load and display the uploaded image
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image.", use_container_width=True)

        # Preprocess and predict the uploaded image
        image = np.array(image)
        predict_image(image)
