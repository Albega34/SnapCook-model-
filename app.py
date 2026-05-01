import os
import io
import numpy as np
import requests
from PIL import Image
from flask import Flask, request, jsonify
from flask_cors import CORS

# Load TFLite models using either tflite_runtime or full tensorflow
try:
    import tflite_runtime.interpreter as tflite
except ImportError:
    try:
        from tensorflow import lite as tflite
    except ImportError:
        import tensorflow as tf
        tflite = tf.lite

app = Flask(__name__)
# Enable CORS for cross-origin requests from the frontend
CORS(app)

# ==========================================
# Paths to the models
# ==========================================
# Look for models in the current directory
BASE_DIR = os.path.abspath(os.path.dirname(__file__))
FOOD_VS_NONFOOD_MODEL_PATH = os.path.join(BASE_DIR, 'food_vs_nonfood.tflite')
SNAPCOOK_MODEL_PATH = os.path.join(BASE_DIR, 'snapcook.tflite')

# Global variables for TFLite interpreters
interpreter_food_vs_nonfood = None
interpreter_snapcook = None
input_details_food_vs_nonfood = None
output_details_food_vs_nonfood = None
input_details_snapcook = None
output_details_snapcook = None


def load_models():
    """Load the TFLite models on app startup."""
    global interpreter_food_vs_nonfood, interpreter_snapcook
    global input_details_food_vs_nonfood, output_details_food_vs_nonfood
    global input_details_snapcook, output_details_snapcook

    if not os.path.exists(FOOD_VS_NONFOOD_MODEL_PATH):
        raise FileNotFoundError(f"Missing food_vs_nonfood.tflite at {FOOD_VS_NONFOOD_MODEL_PATH}")
    if not os.path.exists(SNAPCOOK_MODEL_PATH):
        raise FileNotFoundError(f"Missing snapcook.tflite at {SNAPCOOK_MODEL_PATH}")

    # Load Food vs Non-Food Model
    interpreter_food_vs_nonfood = tflite.Interpreter(model_path=FOOD_VS_NONFOOD_MODEL_PATH)
    interpreter_food_vs_nonfood.allocate_tensors()
    input_details_food_vs_nonfood = interpreter_food_vs_nonfood.get_input_details()
    output_details_food_vs_nonfood = interpreter_food_vs_nonfood.get_output_details()

    # Load SnapCook classification Model
    interpreter_snapcook = tflite.Interpreter(model_path=SNAPCOOK_MODEL_PATH)
    interpreter_snapcook.allocate_tensors()
    input_details_snapcook = interpreter_snapcook.get_input_details()
    output_details_snapcook = interpreter_snapcook.get_output_details()

    print("✅ Models loaded successfully!")


# Class labels exactly matching the original notebook
binary_classes = ['food', 'nonfood']

snapcook_classes = [
    'Baked Potato', 'Crispy Chicken', 'Donut', 'Fries', 'Hot Dog', 'Sandwich',
    'Taco', 'Taquito', 'apple_pie', 'avocado_toast', 'burger', 'burrito',
    'butter_naan', 'chai', 'chapati', 'cheesecake', 'chicken_curry',
    'chole_bhature', 'croissant', 'dal_makhani', 'falafel', 'fried_rice',
    'grilled_salmon_fillet', 'gulab_jamun', 'ice_cream', 'idli', 'jalebi',
    'kaathi_rolls', 'kadai_paneer', 'khichdi', 'macarons', 'malaikofta',
    'masala_dosa', 'momos', 'okonomiyaki', 'omelette', 'onigiri', 'pakode',
    'pancake', 'pani_puri', 'pav_bhaji', 'pizza', 'poke_bowl', 'ramen',
    'samosa', 'spaghetti', 'steak_(grilled)', 'sushi', 'vadapav', 'waffles'
]


def preprocess_image(image_bytes):
    """
    Load image from bytes, convert to RGB, resize to 224x224, 
    and normalize input exactly like the notebook.
    """
    img = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    img = img.resize((224, 224))
    
    img_array = np.array(img, dtype=np.float32)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0  # Normalize to [0, 1] exactly as per the notebook

    return img_array


def run_prediction(img_data):
    """Run inference using both TFLite models."""
    # Step 1: Food vs Non-Food
    interpreter_food_vs_nonfood.set_tensor(input_details_food_vs_nonfood[0]['index'], img_data)
    interpreter_food_vs_nonfood.invoke()
    binary_predictions_raw = interpreter_food_vs_nonfood.get_tensor(output_details_food_vs_nonfood[0]['index'])
    binary_score = binary_predictions_raw[0]

    binary_index = np.argmax(binary_score)
    binary_class = binary_classes[binary_index]
    binary_confidence = 100 * np.max(binary_score)

    # Apply confidence-switch logic exactly as found in the original notebook
    if binary_confidence < 85.0:
        other_index = 1 - binary_index
        binary_class = binary_classes[other_index]
        binary_confidence = 100 * binary_score[other_index]

    # Stop and return if classified as non-food
    if binary_class == 'nonfood':
        return {
            'stage': 'Food vs Non-Food',
            'prediction': binary_class,
            'confidence': float(f"{binary_confidence:.2f}")
        }

    # Step 2: SnapCook Classification (Food identification)
    interpreter_snapcook.set_tensor(input_details_snapcook[0]['index'], img_data)
    interpreter_snapcook.invoke()
    snap_predictions_raw = interpreter_snapcook.get_tensor(output_details_snapcook[0]['index'])
    snap_score = snap_predictions_raw[0]

    snap_index = np.argmax(snap_score)
    snap_class = snapcook_classes[snap_index]
    snap_confidence = 100 * np.max(snap_score)

    return {
        'stage': 'SnapCook Classification',
        'prediction': snap_class,
        'confidence': float(f"{snap_confidence:.2f}")
    }


@app.route('/')
def home():
    return jsonify({
        'status': 'online',
        'message': '🚀 SnapCook Prediction API is running perfectly!',
        'available_endpoints': {
            '/predict': 'POST endpoint accepting image_url in JSON or multipart/form-data for image uploads'
        }
    })


@app.route('/predict', methods=['POST'])
def predict():
    image_bytes = None

    # Option A: JSON input with image_url
    if request.is_json:
        data = request.get_json()
        if not data or 'image_url' not in data:
            return jsonify({'error': 'Missing image_url field in JSON payload'}), 400
        
        image_url = data['image_url']
        try:
            response = requests.get(image_url, timeout=15)
            response.raise_for_status()
            image_bytes = response.content
        except requests.exceptions.RequestException as e:
            return jsonify({'error': f"Error fetching image from URL: {e}"}), 400

    # Option B: Multipart Form Data with image file upload
    elif 'file' in request.files:
        uploaded_file = request.files['file']
        if uploaded_file.filename == '':
            return jsonify({'error': 'Empty filename uploaded'}), 400
        image_bytes = uploaded_file.read()

    # Option C: Direct Image file raw data (binary upload)
    elif request.data:
        image_bytes = request.data

    else:
        return jsonify({'error': 'No image provided. Please send image_url via JSON or upload a file via multipart form-data (key: "file")'}), 400

    try:
        # Preprocess the image
        img_data = preprocess_image(image_bytes)
        
        # Run TFLite inference
        prediction_result = run_prediction(img_data)
        
        return jsonify(prediction_result)

    except Exception as e:
        return jsonify({'error': str(e)}), 500


# Call model loader
load_models()

if __name__ == '__main__':
    # Default Flask execution
    # Port is set to 8000, suitable for typical ngrok mapping or internal testing
    app.run(host='0.0.0.0', port=8000, debug=False)
