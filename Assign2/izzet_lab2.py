# izzet_lab2.py
# COMP 263 - Deep Learning | Lab Assignment 2: Autoencoders and Transfer Learning
# Izzet Abidi | 300898230
# Centennial College | Winter 2026

# ============================================================
# Imports
# ============================================================
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import (Conv2D, Conv2DTranspose, Flatten,
                                      Dense, Input)
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
print("=" * 50)
print("STEP A: DATA LOADING")
print("=" * 50)

(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

# First 60,000 samples for unsupervised learning (images only)
unsupervised_izzet = {'images': train_images}

# Next 10,000 samples for supervised learning (images and labels)
supervised_izzet = {'images': test_images, 'labels': test_labels}

# ============================================================
# Step b: Data Preprocessing
# ============================================================
print("\n" + "=" * 50)
print("STEP B: DATA PREPROCESSING")
print("=" * 50)

# b.1 - Normalize pixel values to [0, 1]
unsupervised_izzet['images'] = unsupervised_izzet['images'].astype('float32') / 255.0
supervised_izzet['images'] = supervised_izzet['images'].astype('float32') / 255.0

# Reshape to add channel dimension for Conv2D layers
unsupervised_izzet['images'] = unsupervised_izzet['images'].reshape(-1, 28, 28, 1)
supervised_izzet['images'] = supervised_izzet['images'].reshape(-1, 28, 28, 1)

# b.2 - One-hot encode supervised labels
supervised_izzet['labels'] = to_categorical(supervised_izzet['labels'])

# b.3 - Display shapes
print(f"Unsupervised images shape: {unsupervised_izzet['images'].shape}")
print(f"Supervised images shape: {supervised_izzet['images'].shape}")
print(f"Supervised labels shape: {supervised_izzet['labels'].shape}")

# ============================================================
# Step c: Data Preparation (Training, Validation, Testing)
# ============================================================
print("\n" + "=" * 50)
print("STEP C: DATA PREPARATION")
print("=" * 50)

# c.1 - Split unsupervised into training (57,000) and validation (3,000)
unsupervised_train_izzet, unsupervised_val_izzet = train_test_split(
    unsupervised_izzet['images'],
    test_size=3000,
    random_state=30
)

# c.2 - Randomly discard 7,000 from supervised (keep 3,000)
supervised_remaining_images, _, supervised_remaining_labels, _ = train_test_split(
    supervised_izzet['images'],
    supervised_izzet['labels'],
    test_size=7000,
    random_state=30
)

# c.3 - Split remaining 3,000 into train (1800), val (600), test (600)
# First split: 1800 train, 1200 temp
x_train_izzet, x_temp_izzet, y_train_izzet, y_temp_izzet = train_test_split(
    supervised_remaining_images,
    supervised_remaining_labels,
    test_size=1200,
    random_state=30
)

# Second split: 600 val, 600 test
x_val_izzet, x_test_izzet, y_val_izzet, y_test_izzet = train_test_split(
    x_temp_izzet,
    y_temp_izzet,
    test_size=600,
    random_state=30
)

# c.4 - Display shapes
print(f"Unsupervised train: {unsupervised_train_izzet.shape}")
print(f"Unsupervised val: {unsupervised_val_izzet.shape}")
print(f"Supervised train: {x_train_izzet.shape} | Labels: {y_train_izzet.shape}")
print(f"Supervised val: {x_val_izzet.shape} | Labels: {y_val_izzet.shape}")
print(f"Supervised test: {x_test_izzet.shape} | Labels: {y_test_izzet.shape}")

# ============================================================
# Step d: Build, Train, and Validate Baseline CNN Model
# ============================================================
print("\n" + "=" * 50)
print("STEP D: BASELINE CNN MODEL")
print("=" * 50)

# d.1 - Build the baseline CNN
cnn_v1_model_izzet = Sequential([
    Conv2D(16, (3, 3), activation='relu', padding='same', strides=2,
           input_shape=(28, 28, 1)),
    Conv2D(8, (3, 3), activation='relu', padding='same', strides=2),
    Flatten(),
    Dense(100),
    Dense(10, activation='softmax')
])

# d.2 - Compile the model
cnn_v1_model_izzet.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# d.3 - Display model summary
print("\nBaseline CNN Summary:")
cnn_v1_model_izzet.summary()

# d.4 - Train and validate
cnn_v1_history_izzet = cnn_v1_model_izzet.fit(
    x_train_izzet, y_train_izzet,
    epochs=10,
    batch_size=256,
    validation_data=(x_val_izzet, y_val_izzet)
)

# ============================================================
# Step e: Test and Analyze Baseline Model
# ============================================================
print("\n" + "=" * 50)
print("STEP E: BASELINE CNN EVALUATION")
print("=" * 50)

# e.1 - Plot Training vs Validation Accuracy
plt.figure(figsize=(10, 6))
plt.plot(cnn_v1_history_izzet.history['accuracy'], color='blue', label='Training Accuracy')
plt.plot(cnn_v1_history_izzet.history['val_accuracy'], color='orange', label='Validation Accuracy')
plt.title('Baseline CNN: Training vs Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

# e.2 - Evaluate on test dataset
v1_test_loss, v1_test_accuracy = cnn_v1_model_izzet.evaluate(x_test_izzet, y_test_izzet)
print(f"\nBaseline CNN Test Accuracy: {v1_test_accuracy:.4f}")

# e.3 - Create predictions
cnn_predictions_izzet = cnn_v1_model_izzet.predict(x_test_izzet)

# e.4 - Confusion Matrix
v1_pred_classes = np.argmax(cnn_predictions_izzet, axis=1)
v1_true_classes = np.argmax(y_test_izzet, axis=1)
v1_cm = confusion_matrix(v1_true_classes, v1_pred_classes)

plt.figure(figsize=(10, 8))
sns.heatmap(v1_cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
plt.title('Baseline CNN: Confusion Matrix')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()

# ============================================================
# Step f: Add Random Noise to Unsupervised Dataset
# ============================================================
print("\n" + "=" * 50)
print("STEP F: NOISE INJECTION")
print("=" * 50)

noise_factor = 0.2

# f.1 - Add Gaussian noise to unsupervised training and validation sets
tf.random.set_seed(30)
x_train_noisy_izzet = unsupervised_train_izzet + noise_factor * tf.random.normal(
    shape=unsupervised_train_izzet.shape
)
x_val_noisy_izzet = unsupervised_val_izzet + noise_factor * tf.random.normal(
    shape=unsupervised_val_izzet.shape
)

# f.2 - Clip values to [0, 1]
x_train_noisy_izzet = tf.clip_by_value(x_train_noisy_izzet, 0.0, 1.0)
x_val_noisy_izzet = tf.clip_by_value(x_val_noisy_izzet, 0.0, 1.0)

print(f"Noisy training set shape: {x_train_noisy_izzet.shape}")
print(f"Noisy validation set shape: {x_val_noisy_izzet.shape}")

# f.3 - Plot first 10 noisy validation images
plt.figure(figsize=(20, 2))
for i in range(10):
    plt.subplot(1, 10, i + 1)
    plt.imshow(x_val_noisy_izzet[i].numpy().reshape(28, 28), cmap='gray')
    plt.xticks([])
    plt.yticks([])
plt.suptitle('First 10 Noisy Validation Images', fontsize=14)
plt.tight_layout()
plt.show()

# ============================================================
# Step g: Build and Pretrain Autoencoder
# ============================================================
print("\n" + "=" * 50)
print("STEP G: DENOISING AUTOENCODER")
print("=" * 50)

# g.1 - Build the autoencoder using Functional API
inputs_izzet = Input(shape=(28, 28, 1))

# Encoder section
e_izzet = Conv2D(16, (3, 3), activation='relu', padding='same', strides=2)(inputs_izzet)
e_izzet = Conv2D(8, (3, 3), activation='relu', padding='same', strides=2)(e_izzet)

# Decoder section
d_izzet = Conv2DTranspose(8, (3, 3), activation='relu', padding='same', strides=2)(e_izzet)
d_izzet = Conv2DTranspose(16, (3, 3), activation='relu', padding='same', strides=2)(d_izzet)
d_izzet = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(d_izzet)

# Create model
autoencoder_izzet = Model(inputs_izzet, d_izzet)

# g.2 - Compile with adam and MSE
autoencoder_izzet.compile(
    optimizer='adam',
    loss='mean_squared_error'
)

# g.3 - Display summary
print("\nAutoencoder Summary:")
autoencoder_izzet.summary()

# g.4 - Train: noisy input -> clean output
autoencoder_izzet.fit(
    x_train_noisy_izzet, unsupervised_train_izzet,
    epochs=10,
    batch_size=256,
    shuffle=True,
    validation_data=(x_val_noisy_izzet, unsupervised_val_izzet)
)

# g.5 - Predict on validation set
autoencoder_predictions_izzet = autoencoder_izzet.predict(x_val_noisy_izzet)

# g.6 - Plot first 10 denoised images
plt.figure(figsize=(20, 2))
for i in range(10):
    plt.subplot(1, 10, i + 1)
    # Use numpy mean to remove the channel axis for plotting
    plt.imshow(np.mean(autoencoder_predictions_izzet[i], axis=-1), cmap='gray')
    plt.xticks([])
    plt.yticks([])
plt.suptitle('First 10 Denoised Validation Images', fontsize=14)
plt.tight_layout()
plt.show()

# ============================================================
# Step h: Build and Perform Transfer Learning on CNN
# ============================================================
print("\n" + "=" * 50)
print("STEP H: TRANSFER LEARNING CNN")
print("=" * 50)

# h.1 - Build CNN using transferred encoder layers from autoencoder
# The encoder layers are already connected to inputs_izzet
# e_izzet holds the output of the last encoder layer
transfer_flatten = Flatten()(e_izzet)
transfer_dense = Dense(100)(transfer_flatten)
transfer_output = Dense(10, activation='softmax')(transfer_dense)

cnn_v2_izzet = Model(inputs_izzet, transfer_output)

# h.2 - Compile the model
cnn_v2_izzet.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# h.3 - Display summary
print("\nPretrained CNN Summary:")
cnn_v2_izzet.summary()

# h.4 - Train and validate on supervised dataset
cnn_v2_history_izzet = cnn_v2_izzet.fit(
    x_train_izzet, y_train_izzet,
    epochs=10,
    batch_size=256,
    validation_data=(x_val_izzet, y_val_izzet)
)

# ============================================================
# Step i: Test and Analyze Pretrained CNN Model
# ============================================================
print("\n" + "=" * 50)
print("STEP I: PRETRAINED CNN EVALUATION")
print("=" * 50)

# i.1 - Plot Training vs Validation Accuracy
plt.figure(figsize=(10, 6))
plt.plot(cnn_v2_history_izzet.history['accuracy'], color='blue', label='Training Accuracy')
plt.plot(cnn_v2_history_izzet.history['val_accuracy'], color='orange', label='Validation Accuracy')
plt.title('Pretrained CNN: Training vs Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

# i.2 - Evaluate on test dataset
v2_test_loss, v2_test_accuracy = cnn_v2_izzet.evaluate(x_test_izzet, y_test_izzet)
print(f"\nPretrained CNN Test Accuracy: {v2_test_accuracy:.4f}")

# i.3 - Create predictions
cnn_v2_predictions_izzet = cnn_v2_izzet.predict(x_test_izzet)

# i.4 - Confusion Matrix
v2_pred_classes = np.argmax(cnn_v2_predictions_izzet, axis=1)
v2_true_classes = np.argmax(y_test_izzet, axis=1)
v2_cm = confusion_matrix(v2_true_classes, v2_pred_classes)

plt.figure(figsize=(10, 8))
sns.heatmap(v2_cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
plt.title('Pretrained CNN: Confusion Matrix')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()

# ============================================================
# Comparison: Baseline vs Pretrained
# ============================================================
print("\n" + "=" * 50)
print("MODEL COMPARISON")
print("=" * 50)

# Validation accuracy comparison plot
plt.figure(figsize=(10, 6))
plt.plot(cnn_v1_history_izzet.history['val_accuracy'], color='red', label='Baseline CNN')
plt.plot(cnn_v2_history_izzet.history['val_accuracy'], color='green', label='Pretrained CNN')
plt.title('Validation Accuracy: Baseline vs Pretrained CNN')
plt.xlabel('Epoch')
plt.ylabel('Validation Accuracy')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

# Test accuracy comparison
print(f"Baseline CNN Test Accuracy:   {v1_test_accuracy:.4f}")
print(f"Pretrained CNN Test Accuracy: {v2_test_accuracy:.4f}")
print(f"Improvement:                  {v2_test_accuracy - v1_test_accuracy:+.4f}")
