import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import numpy as np
import os
from tensorflow.keras.preprocessing.image import load_img, img_to_array

# Step 1: Dataset Preparation
def load_images_from_folder(folder_path, img_size):
    images, labels = [], []
    class_names = sorted(os.listdir(folder_path))
    for label, class_name in enumerate(class_names):
        class_folder = os.path.join(folder_path, class_name)
        if os.path.isdir(class_folder):
            for img_name in sorted(os.listdir(class_folder)):
                img_path = os.path.join(class_folder, img_name)
                img = load_img(img_path, target_size=(img_size, img_size))
                img_array = img_to_array(img) / 255.0
                images.append(img_array)
                labels.append(label)
    return np.array(images), np.array(labels), class_names

# Define paths to the dataset
dataset_folder = r"/Users/riteshchandra/Desktop/Course Work /Course Work - Semester - III/AIT636/Cat-Dog"
train_folder = os.path.join(dataset_folder, "train")
test_folder = os.path.join(dataset_folder, "test")
img_size = 112

# Handle SSL certificate issues
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

# Load training and testing data
print("Loading training data...")
train_images, train_labels, class_names = load_images_from_folder(train_folder, img_size)
print("Loading testing data...")
test_images, test_labels, _ = load_images_from_folder(test_folder, img_size)

print(f"Loaded {len(train_images)} training images and {len(test_images)} testing images.")
print(f"Classes: {class_names}")

# Step 2: Visualize Data
plt.figure(figsize=(10, 10))
for i in range(25):
    plt.subplot(5, 5, i + 1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(train_images[i])
    plt.xlabel(class_names[train_labels[i]])
plt.show()

# Step 3: Define the Model
def build_model(img_size, conv_layers, num_classes):
    model = keras.models.Sequential()
    model.add(keras.layers.InputLayer(input_shape=(img_size, img_size, 3)))

    # Add convolutional layers
    for _ in range(conv_layers):
        model.add(keras.layers.Conv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu'))
        model.add(keras.layers.MaxPooling2D(pool_size=(2, 2), padding='same'))

    # Add dense layers
    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(units=64, activation='relu'))
    model.add(keras.layers.Dense(units=num_classes, activation='softmax'))
    return model

# Parameters
img_sizes = [32, 64, 112]
conv_layers_list = [1, 2, 3]
num_epochs = 15
batch_size = 160

# Step 4: Iterate over configurations and train models
for img_size in img_sizes:
    resized_train_images = np.array([tf.image.resize(img, (img_size, img_size)).numpy() for img in train_images])
    resized_test_images = np.array([tf.image.resize(img, (img_size, img_size)).numpy() for img in test_images])

    for conv_layers in conv_layers_list:
        print(f"\nTraining model with {conv_layers} conv layers and image size {img_size}x{img_size}...")

        # Build and compile the model
        model = build_model(img_size, conv_layers, len(class_names))
        model.compile(optimizer='adam',
                      loss=keras.losses.SparseCategoricalCrossentropy(),
                      metrics=['accuracy'])

        # Display the model summary
        print("Model Summary:")
        model.summary()

        # Train the model
        history = model.fit(
            x=resized_train_images,
            y=train_labels,
            batch_size=batch_size,
            epochs=num_epochs,
            validation_data=(resized_test_images, test_labels),
            verbose=1
        )

        # Step 5: Plot training vs validation accuracy
        plt.plot(history.history['accuracy'], label='Training Accuracy')
        plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
        plt.title(f"Conv Layers: {conv_layers}, Img Size: {img_size}")
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.show()  # Display the graph
        plt.savefig(f"visual_domain_{conv_layers}_conv_{img_size}_img.png")
        plt.clf()

