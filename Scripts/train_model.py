import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint

# Set paths
data_dir = "Data"  # Replace with the parent directory of your dataset
model_save_path = "Model/keras_model.keras"  # Use .keras extension
label_save_path = "Model/labels.txt"

# Ensure necessary directories exist
os.makedirs("Model", exist_ok=True)

# Image parameters
img_size = 300
batch_size = 32

# Preprocessing data using ImageDataGenerator
datagen = ImageDataGenerator(
    rescale=1.0/255.0,  # Normalize pixel values
    validation_split=0.2  # Split data into 80% training and 20% validation
)

train_data = datagen.flow_from_directory(
    data_dir,
    target_size=(img_size, img_size),
    batch_size=batch_size,
    class_mode='categorical',
    subset='training'
)

val_data = datagen.flow_from_directory(
    data_dir,
    target_size=(img_size, img_size),
    batch_size=batch_size,
    class_mode='categorical',
    subset='validation'
)

# Save class labels to a file
class_labels = list(train_data.class_indices.keys())
with open(label_save_path, 'w') as f:
    for label in class_labels:
        f.write(f"{label}\n")

print(f"Class labels: {class_labels}")

# Define the CNN model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(img_size, img_size, 3)),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(len(class_labels), activation='softmax')  # Output layer for classification
])

# Compile the model
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Define a callback to save the best model
checkpoint = ModelCheckpoint(
    model_save_path,  # Ensure this ends with .keras
    monitor='val_accuracy',
    save_best_only=True,
    verbose=1
)

# Train the model
epochs = 20
history = model.fit(
    train_data,
    validation_data=val_data,
    epochs=epochs,
    callbacks=[checkpoint],
    verbose=1
)

# Save the final model
model.save(model_save_path)
print(f"Model saved to {model_save_path}")
