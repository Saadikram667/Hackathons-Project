import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.applications.efficientnet import preprocess_input
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import os
import json
import numpy as np
from sklearn.utils.class_weight import compute_class_weight

# ==================================
# SETTINGS
# ==================================
train_path = r"C:\Users\DELL\OneDrive\Desktop\python_project\BrainTumorDataset\dataset\train"
val_path = r"C:\Users\DELL\OneDrive\Desktop\python_project\BrainTumorDataset\dataset\validation"
IMG_SIZE = 224
BATCH_SIZE = 16
EPOCHS = 20
MODEL_DIR = "saved_models"
os.makedirs(MODEL_DIR, exist_ok=True)

# ==================================
# DATA GENERATORS
# ==================================
train_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    rotation_range=20, zoom_range=0.2, width_shift_range=0.1,
    height_shift_range=0.1, horizontal_flip=True, fill_mode="nearest"
)

val_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

train_data = train_datagen.flow_from_directory(
    train_path, target_size=(IMG_SIZE, IMG_SIZE), batch_size=BATCH_SIZE, class_mode="categorical"
)
val_data = val_datagen.flow_from_directory(
    val_path, target_size=(IMG_SIZE, IMG_SIZE), batch_size=BATCH_SIZE, class_mode="categorical", shuffle=False
)

# Save class indices to JSON for the testing script
with open('class_indices.json', 'w') as f:
    json.dump(train_data.class_indices, f)

# ==================================
# CLASS WEIGHTS (FIX FOR IMBALANCE)
# ==================================
labels = train_data.classes
weights = compute_class_weight(class_weight='balanced', classes=np.unique(labels), y=labels)
class_weight_dict = dict(enumerate(weights))
print("Class Weights applied:", class_weight_dict)

# ==================================
# MODEL ARCHITECTURE
# ==================================
base_model = EfficientNetB0(weights="imagenet", include_top=False, input_shape=(IMG_SIZE, IMG_SIZE, 3))
base_model.trainable = False 

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(256, activation="relu")(x)
x = Dropout(0.5)(x)
predictions = Dense(len(train_data.class_indices), activation="softmax")(x)

model = Model(inputs=base_model.input, outputs=predictions)
model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

# ==================================
# TRAINING
# ==================================
checkpoint = ModelCheckpoint(os.path.join(MODEL_DIR, "best_brain_tumor_model.keras"), 
                             monitor="val_accuracy", save_best_only=True, verbose=1)

model.fit(
    train_data,
    validation_data=val_data,
    epochs=EPOCHS,
    class_weight=class_weight_dict, # Key fix for accuracy
    callbacks=[EarlyStopping(patience=3, restore_best_weights=True), checkpoint]
)

print("Training finished. 'class_indices.json' and 'best_brain_tumor_model.keras' saved.")