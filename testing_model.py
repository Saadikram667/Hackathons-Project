import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.efficientnet import preprocess_input
import numpy as np
import os
import json
import csv

# ==================================
# CONFIGURATION
# ==================================
MODEL_PATH = "saved_models/best_brain_tumor_model.keras"
JSON_PATH = "class_indices.json"
TEST_FOLDER = r"C:\Users\DELL\OneDrive\Desktop\python_project\BrainTumorDataset\New2"

# 1. LOAD MODEL
print("Loading model... (This might take a few seconds)")
model = load_model(MODEL_PATH)

# 2. LOAD CLASS MAPPING (Ensures labels match training exactly)
with open(JSON_PATH, 'r') as f:
    class_indices = json.load(f)
# Invert the mapping: {0: "glioma", 1: "meningioma", ...}
labels = {v: k for k, v in class_indices.items()}

# ==================================
# SETUP CSV EXPORT
# ==================================
csv_filename = "detailed_test_results.csv"
csv_file = open(csv_filename, mode="w", newline="", encoding="utf-8")
csv_writer = csv.writer(csv_file)
# Write the header row for your spreadsheet
csv_writer.writerow(["Image Name", "True Label", "Predicted Label", "Confidence (%)", "Result"])

# ==================================
# TESTING LOOP
# ==================================
correct = 0
total = 0

print(f"\nStarting evaluation on {TEST_FOLDER}...")
print("=" * 60)

for img_name in os.listdir(TEST_FOLDER):
    img_path = os.path.join(TEST_FOLDER, img_name)
    
    try:
        # Load and preprocess image
        img = image.load_img(img_path, target_size=(224, 224))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = preprocess_input(img_array)

        # Predict
        pred = model.predict(img_array, verbose=0)
        predicted_idx = np.argmax(pred)
        predicted_class = labels[predicted_idx]
        confidence = np.max(pred) * 100

        # Determine True Label using the CORRECT abbreviations
        name_lower = img_name.lower()
        
        if "gl" in name_lower or "glioma" in name_lower:    # Changed 'ga' to 'gl'
            true_class = "glioma"
        elif "me" in name_lower or "meningioma" in name_lower: # Changed 'mi' to 'me'
            true_class = "meningioma"
        elif "no" in name_lower or "notumor" in name_lower:    # Changed 'nt' to 'no'
            true_class = "notumor"
        elif "pi" in name_lower or "pituitary" in name_lower:
            true_class = "pituitary"
        else:
            true_class = "unknown"

        # Calculate accuracy
        if predicted_class == true_class:
            correct += 1
            status = "Correct"
        else:
            status = "Incorrect"
            
        total += 1

        # Save this image's result to the CSV spreadsheet
        csv_writer.writerow([img_name, true_class, predicted_class, f"{confidence:.2f}", status])

        # Print to terminal so you can watch it run
        print(f"File: {img_name} | Pred: {predicted_class} | True: {true_class} | Conf: {confidence:.2f}% | [{status}]")

    except Exception as e:
        print(f"Error processing {img_name}: {e}")

# CLOSE THE CSV FILE WHEN FINISHED
csv_file.close()

# ==================================
# FINAL RESULTS
# ==================================
if total > 0:
    accuracy = (correct / total) * 100
    print("\n" + "="*40)
    print("TESTING COMPLETE!")
    print(f"TOTAL IMAGES : {total}")
    print(f"CORRECT      : {correct}")
    print(f"ACCURACY     : {accuracy:.2f}%")
    print(f"Detailed results saved to: {csv_filename}")
    print("="*40)
else:
    print("\nNo images found to test. Please check your TEST_FOLDER path.")