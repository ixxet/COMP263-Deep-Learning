# izzet_lab3.py
# COMP 263 - Deep Learning | Lab Assignment 3: Variational Autoencoders
# Izzet Abidi | 300898230
# Centennial College | Winter 2026

# ============================================================
# Imports
# ============================================================
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (Conv2D, Conv2DTranspose, Flatten,
                                      Dense, Input, Reshape, Layer)

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

# Store 60,000 training samples with images and labels
train_izzet = {'images': train_images, 'labels': train_labels}

# Store 10,000 test samples with images and labels
test_izzet = {'images': test_images, 'labels': test_labels}

# ============================================================
# Step b: Data Preprocessing
# ============================================================
print("\n" + "=" * 50)
print("STEP B: DATA PREPROCESSING")
print("=" * 50)

# b.1 - Normalize pixel values to [0, 1] and reshape for Conv2D
train_izzet['images'] = train_izzet['images'].astype('float32') / 255.0
test_izzet['images'] = test_izzet['images'].astype('float32') / 255.0

train_izzet['images'] = train_izzet['images'].reshape(-1, 28, 28, 1)
test_izzet['images'] = test_izzet['images'].reshape(-1, 28, 28, 1)

# b.2 - Display shapes
print(f"Train images shape: {train_izzet['images'].shape}")
print(f"Test images shape: {test_izzet['images'].shape}")

# ============================================================
# Step c.1: Custom SampleLayer (Reparameterization Trick)
# ============================================================
print("\n" + "=" * 50)
print("STEP C.1: CUSTOM SAMPLE LAYER")
print("=" * 50)


class SampleLayer(Layer):
    """Custom layer implementing the reparameterization trick.
    Takes [z_mu, z_log_sigma] as input and outputs sampled latent vector z.
    z = mu + exp(log_sigma) * epsilon, where epsilon ~ N(0, I)
    """

    def call(self, inputs):
        z_mu, z_log_sigma = inputs

        # Calculate batch size and dimension from input
        batch = tf.shape(z_mu)[0]
        dim = tf.shape(z_mu)[1]

        # Generate random noise from standard normal distribution
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))

        # Reparameterization trick: z = mu + exp(log_sigma) * epsilon
        z = z_mu + tf.exp(z_log_sigma) * epsilon

        return z


print("SampleLayer defined successfully.")

# ============================================================
# Step c.2: Build Encoder
# ============================================================
print("\n" + "=" * 50)
print("STEP C.2: ENCODER")
print("=" * 50)

# Latent dimension
latent_dim = 2

# Input layer
input_img = Input(shape=(28, 28, 1))

# Layer 1: Conv2D with 32 kernels, 3x3, relu, same padding
x = Conv2D(32, (3, 3), activation='relu', padding='same')(input_img)

# Layer 2: Conv2D with 64 kernels, 3x3, relu, same padding, stride 2
x = Conv2D(64, (3, 3), activation='relu', padding='same', strides=2)(x)

# Layer 3: Conv2D with 64 kernels, 3x3, relu, same padding
x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)

# Layer 4: Conv2D with 64 kernels, 3x3, relu, same padding
x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)

# Flatten and Dense layer
x = Flatten()(x)

# Layer 5: Dense with 32 neurons, relu activation
x = Dense(32, activation='relu')(x)

# Latent space parameters
z_mu_izzet = Dense(latent_dim, name='z_mu')(x)
z_log_sigma_izzet = Dense(latent_dim, name='z_log_sigma')(x)

# Output: sample from latent distribution
z_izzet = SampleLayer()([z_mu_izzet, z_log_sigma_izzet])

# Build encoder model
encoder_izzet = Model(input_img, [z_mu_izzet, z_log_sigma_izzet, z_izzet],
                      name='encoder')

print("\nEncoder Summary:")
encoder_izzet.summary()
