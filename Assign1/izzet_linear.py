# izzet_linear.py
# COMP 263 - Deep Learning | Lab Assignment 1: CNNs and RNNs
# Izzet Abidi | 300898230
# Centennial College | Winter 2026

# ============================================================
# Imports
# ============================================================
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

# ============================================================
# Fashion MNIST class labels
# ============================================================
CLASS_NAMES = [
    'T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
    'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot'
]

# ============================================================
# Step a: Get the data
# ============================================================
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

train_izzet = {'images': train_images, 'labels': train_labels}
test_izzet = {'images': test_images, 'labels': test_labels}

# ============================================================
# Step b: Initial Exploration
# ============================================================
print("=" * 50)
print("STEP B: INITIAL EXPLORATION")
print("=" * 50)
print(f"Training dataset size: {train_izzet['images'].shape[0]}")
print(f"Testing dataset size: {test_izzet['images'].shape[0]}")
print(f"Image resolution: {train_izzet['images'].shape[1:]}")
print(f"Maximum pixel value: {np.amax(train_izzet['images'])}")

# ============================================================
# Step c: Data Preprocessing
# ============================================================
print("\n" + "=" * 50)
print("STEP C: DATA PREPROCESSING")
print("=" * 50)

# c.1 - Normalize pixel values to [0, 1]
max_pixel = np.amax(train_izzet['images'])
train_izzet['images'] = train_izzet['images'] / max_pixel
test_izzet['images'] = test_izzet['images'] / max_pixel

# c.2 - One-hot encode the labels
train_izzet['labels'] = to_categorical(train_izzet['labels'])
test_izzet['labels'] = to_categorical(test_izzet['labels'])

# c.3 - Display label shapes
print(f"Training labels shape: {train_izzet['labels'].shape}")
print(f"Testing labels shape: {test_izzet['labels'].shape}")
print(f"Number of classes: {train_izzet['labels'].shape[1]}")

# ============================================================
# Step d: Visualization
# ============================================================

# d.1 - Function to display an image with its true label
def plot_image(image, label):
    """Displays a single image with its class label."""
    plt.imshow(image, cmap='gray')
    plt.title(CLASS_NAMES[np.argmax(label)])
    plt.xticks([])
    plt.yticks([])

# d.2 - Plot first 12 training samples in 4x3 grid
plt.figure(figsize=(8, 8))
for i in range(12):
    plt.subplot(4, 3, i + 1)
    plot_image(train_izzet['images'][i], train_izzet['labels'][i])
plt.suptitle("First 12 Training Samples", fontsize=14)
plt.tight_layout()
plt.show()

# ============================================================
# Step e: Training Data Preparation
# ============================================================
print("\n" + "=" * 50)
print("STEP E: TRAINING DATA PREPARATION")
print("=" * 50)

# Split training data: 80% train, 20% validation, seed = 30
x_train_izzet, x_val_izzet, y_train_izzet, y_val_izzet = train_test_split(
    train_izzet['images'],
    train_izzet['labels'],
    test_size=0.2,
    random_state=30
)

print(f"Training set: {x_train_izzet.shape[0]} samples")
print(f"Validation set: {x_val_izzet.shape[0]} samples")

# ============================================================
# Step f: Build, Train, and Validate CNN Model
# ============================================================
print("\n" + "=" * 50)
print("STEP F: CNN MODEL")
print("=" * 50)

# Reshape images for CNN: add channel dimension (28, 28) -> (28, 28, 1)
x_train_cnn = x_train_izzet.reshape(-1, 28, 28, 1)
x_val_cnn = x_val_izzet.reshape(-1, 28, 28, 1)
x_test_cnn = test_izzet['images'].reshape(-1, 28, 28, 1)

# f.1 - Build the CNN model
cnn_model_izzet = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    MaxPooling2D((2, 2)),
    Conv2D(32, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(100),
    Dense(10, activation='softmax')
])

# f.2 - Compile the model
cnn_model_izzet.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# f.3 - Display model summary
print("\nCNN Model Summary:")
cnn_model_izzet.summary()

# f.4 - Train and validate the model
cnn_history_izzet = cnn_model_izzet.fit(
    x_train_cnn, y_train_izzet,
    epochs=8,
    batch_size=256,
    validation_data=(x_val_cnn, y_val_izzet)
)

# ============================================================
# Step g: Test and Analyze CNN Model
# ============================================================
print("\n" + "=" * 50)
print("STEP G: CNN EVALUATION")
print("=" * 50)

# g.1 - Plot Training vs Validation Accuracy
plt.figure(figsize=(10, 6))
plt.plot(cnn_history_izzet.history['accuracy'], color='blue', label='Training Accuracy')
plt.plot(cnn_history_izzet.history['val_accuracy'], color='orange', label='Validation Accuracy')
plt.title('CNN Model: Training vs Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

# g.2 - Evaluate on test dataset
cnn_test_loss, cnn_test_accuracy = cnn_model_izzet.evaluate(x_test_cnn, test_izzet['labels'])
print(f"\nCNN Test Accuracy: {cnn_test_accuracy:.4f}")

# g.3 - Create predictions on test dataset
cnn_predictions_izzet = cnn_model_izzet.predict(x_test_cnn)

# g.4 - Function to plot prediction probability distribution
def plot_probability_distribution(true_label, probabilities):
    """Plots the probability distribution of predictions as a histogram.

    Green bars indicate the true label, blue bars indicate all other classes.
    """
    true_index = np.argmax(true_label)
    colors = ['green' if i == true_index else 'blue' for i in range(len(probabilities))]

    plt.bar(range(len(probabilities)), probabilities, color=colors)
    plt.xticks(range(len(CLASS_NAMES)), CLASS_NAMES, rotation=45, ha='right', fontsize=8)
    plt.xlabel('Class')
    plt.ylabel('Probability')
    plt.ylim(0, 1)

# g.5 - Display 4 test images (starting from index 30) with probability distributions
# Student ID last 2 digits = 30, so display images 30, 31, 32, 33
start_index = 30
plt.figure(figsize=(16, 8))
for i in range(4):
    idx = start_index + i

    # Image subplot
    plt.subplot(2, 4, i + 1)
    plt.imshow(test_izzet['images'][idx], cmap='gray')
    predicted_label = CLASS_NAMES[np.argmax(cnn_predictions_izzet[idx])]
    true_label = CLASS_NAMES[np.argmax(test_izzet['labels'][idx])]
    plt.title(f"True: {true_label}\nPred: {predicted_label}", fontsize=9)
    plt.xticks([])
    plt.yticks([])

    # Probability distribution subplot
    plt.subplot(2, 4, i + 5)
    plot_probability_distribution(test_izzet['labels'][idx], cnn_predictions_izzet[idx])

plt.suptitle('CNN Predictions: Images and Probability Distributions', fontsize=14)
plt.tight_layout()
plt.show()

# g.6 - Confusion Matrix
cnn_pred_classes = np.argmax(cnn_predictions_izzet, axis=1)
cnn_true_classes = np.argmax(test_izzet['labels'], axis=1)
cnn_cm = confusion_matrix(cnn_true_classes, cnn_pred_classes)

plt.figure(figsize=(10, 8))
sns.heatmap(cnn_cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
plt.title('CNN Model: Confusion Matrix')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()

# ============================================================
# Step h: Build, Train, Validate, Test, and Analyze RNN Model
# ============================================================
# Repeats Steps f and g for an RNN with LSTM architecture.
# Image height (28 rows) serves as the timestep dimension,
# with each row of 28 pixels as the feature vector per step.

from tensorflow.keras.layers import LSTM

print("\n" + "=" * 50)
print("STEP H: RNN (LSTM) MODEL")
print("=" * 50)

# RNN uses (28, 28) input directly — no channel dimension needed
x_train_rnn = x_train_izzet
x_val_rnn = x_val_izzet
x_test_rnn = test_izzet['images']

# h.1 - Build the RNN model
rnn_model_izzet = Sequential([
    LSTM(128, input_shape=(28, 28)),
    Dense(10, activation='softmax')
])

# h.2 - Compile the model
rnn_model_izzet.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# h.3 - Display model summary
print("\nRNN Model Summary:")
rnn_model_izzet.summary()

# h.4 - Train and validate the model
rnn_history_izzet = rnn_model_izzet.fit(
    x_train_rnn, y_train_izzet,
    epochs=8,
    batch_size=256,
    validation_data=(x_val_rnn, y_val_izzet)
)

# ---- RNN Evaluation (repeating Step g for the RNN) ----
print("\n" + "=" * 50)
print("RNN EVALUATION")
print("=" * 50)

# Plot Training vs Validation Accuracy
plt.figure(figsize=(10, 6))
plt.plot(rnn_history_izzet.history['accuracy'], color='blue', label='Training Accuracy')
plt.plot(rnn_history_izzet.history['val_accuracy'], color='orange', label='Validation Accuracy')
plt.title('RNN Model: Training vs Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

# Evaluate on test dataset
rnn_test_loss, rnn_test_accuracy = rnn_model_izzet.evaluate(x_test_rnn, test_izzet['labels'])
print(f"\nRNN Test Accuracy: {rnn_test_accuracy:.4f}")

# Create predictions on test dataset
rnn_predictions_izzet = rnn_model_izzet.predict(x_test_rnn)

# Display 4 test images (starting from index 30) with probability distributions
plt.figure(figsize=(16, 8))
for i in range(4):
    idx = start_index + i

    # Image subplot
    plt.subplot(2, 4, i + 1)
    plt.imshow(test_izzet['images'][idx], cmap='gray')
    predicted_label = CLASS_NAMES[np.argmax(rnn_predictions_izzet[idx])]
    true_label = CLASS_NAMES[np.argmax(test_izzet['labels'][idx])]
    plt.title(f"True: {true_label}\nPred: {predicted_label}", fontsize=9)
    plt.xticks([])
    plt.yticks([])

    # Probability distribution subplot
    plt.subplot(2, 4, i + 5)
    plot_probability_distribution(test_izzet['labels'][idx], rnn_predictions_izzet[idx])

plt.suptitle('RNN Predictions: Images and Probability Distributions', fontsize=14)
plt.tight_layout()
plt.show()

# Confusion Matrix
rnn_pred_classes = np.argmax(rnn_predictions_izzet, axis=1)
rnn_true_classes = np.argmax(test_izzet['labels'], axis=1)
rnn_cm = confusion_matrix(rnn_true_classes, rnn_pred_classes)

plt.figure(figsize=(10, 8))
sns.heatmap(rnn_cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
plt.title('RNN Model: Confusion Matrix')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()

# ============================================================
# Final Comparison
# ============================================================
print("\n" + "=" * 50)
print("MODEL COMPARISON")
print("=" * 50)
print(f"CNN Test Accuracy:  {cnn_test_accuracy:.4f}")
print(f"RNN Test Accuracy:  {rnn_test_accuracy:.4f}")
print(f"Difference:         {abs(cnn_test_accuracy - rnn_test_accuracy):.4f}")
