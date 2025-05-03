import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Load the trained model
model = load_model(
    "C:/Users/CHPde/OneDrive/Bureau/Projects/Sign Language Project/ADS-Project/models/sign_language_model_cropped100.h5"
)

# Initialize webcam
cap = cv2.VideoCapture(0)

# Class labels based on the folder names (adjust these based on your folder structure)
class_labels = [
    "A",
    "B",
    "C",
    "D",
    "E",
    "F",
    "G",
    "H",
    "I",
    "J",
    "K",
    "L",
    "M",
    "N",
    "O",
    "P",
    "Q",
    "R",
    "S",
    "T",
    "U",
    "V",
    "W",
    "X",
    "Y",
    "Z",
    "del",
    "nothing",
    "space",
]

# Initialize written word
written_word = ""

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Get the height and width of the frame
    height, width, _ = frame.shape

    # Calculate the center of the image and size for the square region
    min_side = min(height, width)
    center_x, center_y = width // 2, height // 2

    # Draw a square in the center of the frame (this is the area being fed to the model)
    top_left = (center_x - min_side // 2, center_y - min_side // 2)
    bottom_right = (center_x + min_side // 2, center_y + min_side // 2)

    # Create a stylish frame with a gradient (just for visual appeal)
    cv2.rectangle(frame, top_left, bottom_right, (0, 255, 0), 3)

    # Crop the region inside the square and resize it to match the model input size (100x100)
    cropped_frame = frame[
        center_y - min_side // 2 : center_y + min_side // 2,
        center_x - min_side // 2 : center_x + min_side // 2,
    ]
    resized_frame = cv2.resize(cropped_frame, (100, 100))

    # Preprocess the resized frame (normalize pixel values)
    normalized_frame = resized_frame / 255.0  # Normalize pixel values
    uint8_frame = np.uint8(normalized_frame * 255)  # Convert back to uint8

    # Convert from BGR to RGB (model expects RGB)
    rgb_frame = cv2.cvtColor(uint8_frame, cv2.COLOR_BGR2RGB)

    # Add batch dimension (as model expects a batch)
    model_input = np.expand_dims(rgb_frame, axis=0)

    # Get prediction and confidence for the cropped square frame
    prediction = model.predict(model_input)
    predicted_class = np.argmax(prediction)
    confidence = np.max(prediction)

    # Get the predicted class label
    predicted_label = class_labels[predicted_class]

    # Display the predicted letter and confidence on the webcam feed
    cv2.putText(
        frame,
        f"Predicted: {predicted_label} ({confidence * 100:.2f}%)",  # Show confidence as percentage
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (255, 0, 0),
        2,
    )

    # Show the current word being built (accumulated letters)
    cv2.putText(
        frame,
        f"Current Word: {written_word}",  # Display the written word
        (10, 70),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (255, 255, 255),
        2,
    )

    # Show the live webcam feed with the square preview area and the predicted letter
    cv2.imshow("Sign Language Recognition", frame)

    # Keyboard input to confirm the letter or clear it
    key = cv2.waitKey(1) & 0xFF

    # If 'Enter' is pressed, confirm the letter and add to the word
    if key == 13:  # Enter key
        if (
            predicted_label != "nothing"
            and predicted_label != "del"
            and predicted_label != "space"
        ):
            written_word += predicted_label

    # If 'Backspace' is pressed, delete the last character from the word
    if key == 8:  # Backspace key
        written_word = written_word[:-1]

    # If 'Esc' is pressed, exit the loop
    if key == 27:  # Esc key
        break

cap.release()
cv2.destroyAllWindows()
