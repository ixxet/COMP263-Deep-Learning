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

# ============================================================
# Step c.3: Build Decoder
# ============================================================
print("\n" + "=" * 50)
print("STEP C.3: DECODER")
print("=" * 50)

# Decoder input: latent dimension
decoder_input = Input(shape=(latent_dim,))

# Layer 1: Dense to match encoder's pre-flatten shape (14*14*64 = 12544)
d = Dense(14 * 14 * 64)(decoder_input)

# Layer 2: Reshape back to image-like tensor
d = Reshape((14, 14, 64))(d)

# Layer 3: Transposed convolution to upsample (14x14 -> 28x28)
d = Conv2DTranspose(32, (3, 3), activation='relu', padding='same', strides=2)(d)

# Layer 4: Conv2D with sigmoid to produce output image
d = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(d)

# Build decoder model
decoder_izzet = Model(decoder_input, d, name='decoder_izzet')

print("\nDecoder Summary:")
decoder_izzet.summary()

# ============================================================
# Step c.4: Build VAE
# ============================================================
print("\n" + "=" * 50)
print("STEP C.4: VAE ASSEMBLY")
print("=" * 50)

# Connect encoder input to decoder output
y = decoder_izzet(z_izzet)
vae_izzet = Model(input_img, y, name='vae_izzet')

print("\nVAE Summary:")
vae_izzet.summary()

# ============================================================
# Step d: Define KL Divergence Loss
# ============================================================
print("\n" + "=" * 50)
print("STEP D: KL DIVERGENCE LOSS")
print("=" * 50)

# KL divergence: regularize latent distribution toward standard normal
kl_loss = -0.5 * tf.reduce_mean(
    z_mu_izzet - tf.square(z_mu_izzet) - tf.exp(z_log_sigma_izzet) + 1
)

# Add KL loss to the model
vae_izzet.add_loss(kl_loss)

# Compile with adam optimizer and MSE reconstruction loss
vae_izzet.compile(optimizer='adam', loss='mean_squared_error')

print("KL divergence loss added and model compiled.")

# ============================================================
# Step e: Train the VAE
# ============================================================
print("\n" + "=" * 50)
print("STEP E: TRAINING")
print("=" * 50)

vae_izzet.fit(
    train_izzet['images'], train_izzet['images'],
    epochs=10,
    batch_size=256
)

# ============================================================
# Step f: Generate Samples from the VAE (10x10 Grid)
# ============================================================
print("\n" + "=" * 50)
print("STEP F: SAMPLE GENERATION")
print("=" * 50)

from scipy.stats import norm as scipy_norm

n = 10
figure_size = 28
batch_size = 256

# Create grid of quantile points from standard normal
# Using scipy.stats.norm.ppf (percent point function = quantile/inverse CDF)
grid_x = scipy_norm.ppf(np.linspace(0.05, 0.95, n))
grid_y = scipy_norm.ppf(np.linspace(0.05, 0.95, n))

# Generate images for each grid point
figure = np.zeros((figure_size * n, figure_size * n))
for i, yi in enumerate(grid_x):
    for j, xi in enumerate(grid_y):
        z_sample = np.array([[xi, yi]])
        z_sample = np.tile(z_sample, batch_size).reshape(batch_size, 2)
        x_decoded = decoder_izzet.predict(z_sample, batch_size=batch_size)
        img = x_decoded[0].reshape(figure_size, figure_size)
        figure[i * figure_size: (i + 1) * figure_size,
               j * figure_size: (j + 1) * figure_size] = img

# Plot the generated sample grid
plt.figure(figsize=(20, 20))
plt.imshow(figure, cmap='gray')
plt.title('VAE Generated Samples (10x10 Latent Grid)', fontsize=16)
plt.axis('off')
plt.show()

# ============================================================
# Step g: Display (Plot) Latent Space of z_mu
# ============================================================
print("\n" + "=" * 50)
print("STEP G: LATENT SPACE VISUALIZATION")
print("=" * 50)

# Build model to extract z_mu from the encoder
z_mu_model = Model(input_img, z_mu_izzet, name='z_mu_model')

# Predict latent space encoding of test dataset
z_mu_test = z_mu_model.predict(test_izzet['images'])

# Plot 2D scatter of latent space colored by class label
plt.figure(figsize=(12, 10))
scatter = plt.scatter(
    z_mu_test[:, 0], z_mu_test[:, 1],
    c=test_izzet['labels'],
    cmap='tab10',
    alpha=0.5,
    s=5
)
plt.colorbar(scatter, label='Class Label')
plt.title('VAE Latent Space (z_mu) - Test Set', fontsize=16)
plt.xlabel('z_mu[0]')
plt.ylabel('z_mu[1]')
plt.grid(True, alpha=0.3)
plt.show()

print("\nAssignment 3 complete.")
