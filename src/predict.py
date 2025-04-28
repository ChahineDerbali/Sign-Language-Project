import numpy as np
import cv2


def predict_letter(frame, model):
    # Preprocess frame for model prediction
    frame_resized = cv2.resize(frame, (64, 64)) / 255.0  # Resize and normalize
    frame_input = np.expand_dims(frame_resized, axis=0)  # Add batch dimension

    # Predict with the model
    prediction = model.predict(frame_input)
    predicted_class = np.argmax(prediction)  # Get the index of the predicted class

    # Map to letter
    predicted_letter = chr(predicted_class + ord("A"))  # Convert to letter
    return predicted_letter
