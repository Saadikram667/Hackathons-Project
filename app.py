from flask import Flask, request, jsonify, render_template
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.efficientnet import preprocess_input
import os
import io
from PIL import Image

# 1. Initialize Flask to look in the CURRENT folder for HTML and videos
app = Flask(__name__, template_folder='.', static_folder='', static_url_path='')

# 2. Load Model
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "saved_models", "best_brain_tumor_model.keras")

print("Loading ML model... Please wait.")
model = load_model(MODEL_PATH)
print("Model loaded successfully!")

# Class Labels
labels = {0: 'glioma', 1: 'meningioma', 2: 'notumor', 3: 'pituitary'}

# ==========================================
# 3. WEBSITE ROUTES
# ==========================================

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/page2.html')
def page2():
    return render_template('page2.html')

@app.route('/page3.html')
def page3():
    return render_template('page3.html')

# ==========================================
# 4. API ROUTE (The Brains)
# ==========================================

@app.route('/predict', methods=['POST'])
def predict():
    # 1. Look for 'mri' to match your JavaScript FormData
    if 'mri' not in request.files:
        return jsonify({'error': 'No MRI file uploaded'})
    
    file = request.files['mri']
    
    try:
        # Read the uploaded image in memory
        img = Image.open(io.BytesIO(file.read())).convert('RGB')
        img = img.resize((224, 224))
        
        # Preprocess for EfficientNet
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = preprocess_input(img_array)
        
        # Predict
        pred = model.predict(img_array)
        predicted_idx = np.argmax(pred)
        confidence = float(np.max(pred) * 100)
        result_label = labels[predicted_idx]
        
        # 2. Logic to map the AI result to your UI's expected variables
        is_tumor = (result_label != 'notumor')
        
        if is_tumor:
            risk_level = "CRITICAL" if confidence >= 90 else "ELEVATED"
            formatted_type = result_label.capitalize() # "Glioma", "Meningioma", etc.
        else:
            risk_level = "MINIMAL"
            formatted_type = "None Detected"

        # 3. Send back exactly what your JavaScript is asking for
        return jsonify({
            'tumorDetected': is_tumor,
            'tumorType': formatted_type,
            'confidence': f"{confidence:.2f}%",
            'riskLevel': risk_level
        })
        
    except Exception as e:
        print(f"Error during prediction: {e}")
        return jsonify({'error': 'Failed to process image on server.'})

if __name__ == '__main__':
    app.run(port=5000, debug=True)