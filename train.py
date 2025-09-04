import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.applications import MobileNetV2
import matplotlib.pyplot as plt

# Define the image size and batch size
IMAGE_SIZE = (128, 128)
BATCH_SIZE = 32
NUM_CLASSES = 4 # There are 4 classes in your dataset

# Path to your dataset
# Make sure this matches your folder structure exactly
DATASET_PATH = 'C:/Users/HP/Downloads/archive (1)/Training' 

# Use ImageDataGenerator for data augmentation and loading
datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest',
    validation_split=0.2
)

train_generator = datagen.flow_from_directory(
    DATASET_PATH,
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='training'
)

validation_generator = datagen.flow_from_directory(
    DATASET_PATH,
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='validation'
)

# Load the MobileNetV2 model pre-trained on ImageNet
# We use 'include_top=False' to remove the final classification layer
base_model = MobileNetV2(
    input_shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3),
    include_top=False,
    weights='imagenet'
)

# Freeze the base model to prevent its weights from being updated during training
base_model.trainable = False

# Create a new model on top of the pre-trained model
model = Sequential([
    base_model,
    GlobalAveragePooling2D(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(NUM_CLASSES, activation='softmax')
])

# Compile the new model
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Define a checkpoint to save the best model during training
checkpoint = ModelCheckpoint(
    'brain_model.keras', 
    monitor='val_accuracy', 
    verbose=1, 
    save_best_only=True, 
    mode='max'
)

# Define EarlyStopping to stop training if validation accuracy plateaus
early_stopping = EarlyStopping(
    monitor='val_accuracy', 
    patience=5, 
    mode='max', 
    restore_best_weights=True
)

# Train the model with more epochs and the new callbacks
history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // BATCH_SIZE,
    epochs=50, # You can increase this for better accuracy
    validation_data=validation_generator,
    validation_steps=validation_generator.samples // BATCH_SIZE,
    callbacks=[checkpoint, early_stopping]
)

print("Model training complete and saved as brain_model.keras!")

# --- New code to plot and save the graph ---
def plot_and_save_graph(history):
    """
    Plots the training and validation accuracy and loss and saves the graph as a PNG file.
    """
    # Create a new figure
    plt.figure(figsize=(12, 6))

    # Plot training and validation accuracy
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    plt.savefig('accuracy_graph.png')
    plt.show() # This line displays the plot in a new window

    # Create a new figure for loss
    plt.figure(figsize=(12, 6))
    
    # Plot training and validation loss
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig('loss_graph.png')
    plt.show() # This line displays the plot in a new window

# Call the function after training is complete
plot_and_save_graph(history)
print("Graphs saved as accuracy_graph.png and loss_graph.png!")
