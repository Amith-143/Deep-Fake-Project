import argparse
import cv2
import numpy as np
import os

# Suppress TensorFlow logging to keep terminal clean
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

try:
    import tensorflow as tf
    from tensorflow.keras.models import load_model
    from mtcnn import MTCNN
except ImportError as e:
    print("Error: Missing required libraries. Please run: pip install -r requirements.txt")
    print(e)
    exit(1)


class DeepfakeDetector:

    def __init__(self, model_path, input_shape=(224, 224)):
        self.model_path = model_path
        self.input_shape = input_shape
        
        # Initialize MTCNN for robust face extraction
        self.face_detector = MTCNN()
        
        print(f"[*] Loading model from {model_path}...")
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found at {model_path}. Please provide a valid pre-trained model.")
        
        self.model = load_model(model_path, compile=False)
        print("[*] Model loaded successfully.")

    def extract_face(self, image):

        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Detect faces
        results = self.face_detector.detect_faces(image_rgb)
        
        if not results:
            return None
        
        # We assume the first detected face (usually the most prominent) is our target
        bounding_box = results[0]['box']
        x, y, w, h = bounding_box
        
        # Ensure bounding box coordinates are within image dimensions and positive
        x, y = max(0, x), max(0, y)
        
        # Crop the face from the original BGR image
        cropped_face = image[y:y+h, x:x+w]
        return cropped_face

    def preprocess_image(self, image_path):
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image file not found at {image_path}.")
            
        # Read the image using OpenCV
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not read the image at {image_path}. Ensure it is a valid image format.")

        # Attempt to extract the face
        face = self.extract_face(image)
        if face is None:
            print("[!] Warning: No face detected in the image. Using the entire image as fallback.")
            face = image # Fallback to using the whole image if face detection fails
            
        # Resize to the model's expected input dimensions
        face_resized = cv2.resize(face, self.input_shape)
        
        # Convert BGR (OpenCV) to RGB (most models are trained on RGB)
        face_rgb = cv2.cvtColor(face_resized, cv2.COLOR_BGR2RGB)
        
        # Normalize pixel values to [0, 1] scaling
        # (Change to [-1, 1] if your specific model architecture, like Xception, requires it: (face_rgb / 127.5) - 1.0)
        face_normalized = face_rgb.astype("float32") / 255.0
        
        # Add batch dimension: shape goes from (height, width, channels) to (1, height, width, channels)
        face_batched = np.expand_dims(face_normalized, axis=0)
        
        return face_batched

    def predict(self, image_path):
        """
        Preprocesses the image, runs inference, and prints the result.
        """
        print(f"\n[*] Processing image: {image_path}")
        try:
            preprocessed_input = self.preprocess_image(image_path)
        except Exception as e:
            print(f"[!] Error during preprocessing: {e}")
            return

        print("[*] Running inference...")
        
        # Run inference
        # Assuming binary classification model outputting a single probability (sigmoid activation).
        prediction = self.model.predict(preprocessed_input, verbose=0)
        
        # Extract the probability score
        # Note: If the model outputs 2 units with softmax, you would use prediction[0][1] (assuming index 1 is fake)
        confidence = float(prediction[0][0])
        
        threshold = 0.5
        is_fake = confidence > threshold
        
        # Display results
        print("\n" + "="*30)
        print("       DETECTION RESULTS")
        print("="*30)
        print(f"Deepfake Probability: {confidence * 100:.2f}%")
        
        if is_fake:
            print("\nResult: FAKE")
            print("The model indicates high probability of AI manipulation.")
        else:
            print("\nResult: REAL")
            print("The model indicates this is likely an authentic image.")
        print("="*30 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Deepfake Image Detection using pre-trained Keras models.")
    parser.add_argument("--image", required=True, help="Path to the test image file (.jpg, .png).")
    parser.add_argument("--model", required=True, help="Path to the pre-trained model file (.h5 or .keras).")
    parser.add_argument("--input_size", type=int, default=224, help="Input size expected by the model (default: 224 for 224x224).")
    
    args = parser.parse_args()
    
    
    try:
        # Instantiate detector
        detector = DeepfakeDetector(
            model_path=args.model, 
            input_shape=(args.input_size, args.input_size)
        )
        
        # Run prediction
        detector.predict(image_path=args.image)
        
    except Exception as e:
        print(f"\n[!] A fatal error occurred: {e}")
