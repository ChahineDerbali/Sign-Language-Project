import cv2
import numpy as np
from tensorflow.keras.models import load_model

from predict import predict_letter

# Load the trained model
model = load_model(
    "C:/Users/CHPde/OneDrive/Bureau/Projects/Sign Language Project/ADS-Project/src/sign_language_model.h5"
)

# Initialize webcam
cap = cv2.VideoCapture(0)

# Load pre-trained Haar Cascade for face detection
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Get the height and width of the frame
    height, width, _ = frame.shape

    # Split the frame into two parts (left and right)
    left_half = frame[:, : width // 2]  # Left half for the hand
    right_half = frame[:, width // 2 :]  # Right half for the face

    # Draw a rectangle to show the user the area where the hand should be
    cv2.rectangle(
        frame, (0, 0), (width // 2, height), (0, 255, 0), 2
    )  # Green rectangle on the left

    # Preprocess the left part of the frame (for hand detection)
    left_half_resized = cv2.resize(left_half, (64, 64))  # Resize to model input size
    left_half_normalized = left_half_resized / 255.0  # Normalize pixel values

    # Convert the normalized image back to uint8 (values in the range [0, 255])
    left_half_uint8 = np.uint8(left_half_normalized * 255)

    # Convert from BGR to RGB (now that the image is uint8)
    left_half_rgb = cv2.cvtColor(left_half_uint8, cv2.COLOR_BGR2RGB)

    # Add batch dimension
    left_half_input = np.expand_dims(left_half_rgb, axis=0)

    # Get prediction and confidence for the left half (hand region)
    prediction = model.predict(left_half_input)
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

    # Show the live feed
    cv2.imshow("Sign Language Recognition", frame)

    # Exit on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
