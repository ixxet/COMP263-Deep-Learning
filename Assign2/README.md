# Assignment 2 — Autoencoders and Transfer Learning

**Course:** COMP 263 — Deep Learning
**Weight:** Lab Assignment 2 — Due Week 7
**Student:** Izzet Abidi (300898230)

---

## 1. Overview

This assignment explores **unsupervised feature learning** through denoising autoencoders and **transfer learning** by reusing learned encoder weights for supervised classification. It builds directly on Assignment 1's CNN foundations by introducing the Functional API, encoder-decoder architectures, and the concept of pretraining on unlabeled data to boost performance on limited labeled data. The core competency is understanding how representations learned from reconstruction tasks transfer to classification tasks, and measuring the impact of pretraining versus training from scratch.

---

## 2. Exercise Breakdown

### Exercise 1: Autoencoders and Transfer Learning (100 marks)

**Objective:** Build a denoising autoencoder on unlabeled Fashion MNIST data, transfer the encoder weights to a CNN classifier, and compare its performance against a baseline CNN trained from scratch — both operating on a small labeled subset of just 3,000 samples.

**What the script does:**

1. **Data Loading (Step a)** — Imports `fashion_mnist` from TensorFlow. Stores the first 60,000 samples in `unsupervised_izzet` (images only, no labels — simulating unlabeled data) and the remaining 10,000 samples in `supervised_izzet` (images and labels for classification).

2. **Data Preprocessing (Step b)** — Normalizes all pixel values to [0, 1] by dividing by 255. One-hot encodes the supervised labels using `to_categorical()`. Prints the shapes of `unsupervised_izzet['images']`, `supervised_izzet['images']`, and `supervised_izzet['labels']`.

3. **Data Preparation (Step c)** — Splits the unsupervised dataset into 57,000 training and 3,000 validation samples (`random_state=30`). From the supervised dataset, randomly discards 7,000 samples to simulate a data-scarce scenario, then splits the remaining 3,000 into training (1,800), validation (600), and testing (600) across two sequential splits (`random_state=30`). Prints shapes of all resulting datasets.

4. **Baseline CNN (Steps d–e)** — Builds `cnn_v1_model_izzet` as a `Sequential` CNN:
   - Input: (28, 28, 1)
   - Conv2D(16, 3×3, ReLU, same padding, stride 2)
   - Conv2D(8, 3×3, ReLU, same padding, stride 2)
   - Flatten → Dense(100) → Dense(10, softmax)

   Compiles with Adam optimizer, categorical crossentropy loss, and accuracy metric. Trains for 10 epochs at batch size 256 on the supervised training/validation sets. Evaluates on the test set, generates `cnn_predictions_izzet`, plots training vs. validation accuracy, and renders a confusion matrix heatmap.

5. **Noise Addition (Step f)** — Adds Gaussian noise to the unsupervised training and validation images using `tf.random.normal()` with a noise factor of 0.2 (`seed=30`). Clips noisy values to [0, 1] with `tf.clip_by_value()`. Plots the first 10 noisy validation images.

6. **Denoising Autoencoder (Step g)** — Builds `autoencoder_izzet` using the Functional API (`tf.keras.Model`):
   - **Encoder (`e_izzet`):** Conv2D(16, 3×3, ReLU, same, stride 2) → Conv2D(8, 3×3, ReLU, same, stride 2)
   - **Decoder (`d_izzet`):** Conv2DTranspose(8, 3×3, ReLU, same, stride 2) → Conv2DTranspose(16, 3×3, ReLU, same, stride 2) → Conv2D(1, 3×3, sigmoid, same)

   Compiles with Adam optimizer and mean squared error loss. Trains for 10 epochs at batch size 256 with shuffle enabled, using noisy images as input and clean images as target. Predicts on the validation set and displays the first 10 denoised images.

7. **Transfer Learning CNN (Steps h–i)** — Builds `cnn_v2_izzet` using the Functional API, transferring the autoencoder's input layer and encoder section:
   - Input: transferred from `inputs_izzet`
   - Encoder layers: transferred from `e_izzet`
   - Flatten → Dense(100) → Dense(10, softmax)

   Compiles and trains with the same settings as the baseline. Evaluates on the test set, generates predictions, and renders a confusion matrix. Compares baseline vs. pretrained validation accuracy curves side-by-side and reports both test accuracies.

**Key design decisions:**

- **Unsupervised/supervised split rationale:** Using 60,000 unlabeled samples for autoencoder pretraining and only 3,000 labeled samples for classification simulates a realistic scenario where labeled data is expensive but unlabeled data is abundant.
- **Strided convolutions instead of pooling:** Both the baseline CNN and autoencoder use stride-2 convolutions for downsampling, replacing explicit MaxPooling layers. This learns the downsampling operation rather than using a fixed heuristic.
- **Same padding throughout:** Preserves spatial dimensions before striding, ensuring predictable output sizes through the encoder-decoder pipeline and enabling clean transposed convolution reconstruction.
- **Noise factor 0.2:** A moderate noise level that corrupts images enough to force the autoencoder to learn robust features without destroying structural information entirely.
- **Sigmoid output on autoencoder:** Since input images are normalized to [0, 1], sigmoid ensures reconstructed pixel values stay in the same range, compatible with the MSE loss.
- **Functional API for autoencoder and transfer model:** Required to share layers between the autoencoder and the transfer learning CNN — the Sequential API does not support layer reuse across models.

**File manifest:**

| File | Purpose |
|------|---------|
| `izzet_lab2.py` | Complete implementation — data pipeline, baseline CNN, autoencoder, transfer learning CNN, all evaluation and comparison |

---

## 3. Runbook

### Prerequisites

```bash
$ pip install tensorflow numpy matplotlib seaborn scikit-learn
```

Requires Python 3.8+ and TensorFlow 2.x.

### Execution

```bash
$ cd Assign2/
$ python izzet_lab2.py
```

The script runs sequentially: data loading → preprocessing → data splitting → baseline CNN → noise addition → autoencoder pretraining → transfer learning CNN → comparison. All plots display interactively via Matplotlib.

### Sample Output

```
Unsupervised images shape: (60000, 28, 28, 1)
Supervised images shape: (10000, 28, 28, 1)
Supervised labels shape: (10000, 10)
...
Unsupervised train: (57000, 28, 28, 1)
Unsupervised val: (3000, 28, 28, 1)
Supervised train: (1800, 28, 28, 1) | Labels: (1800, 10)
Supervised val: (600, 28, 28, 1) | Labels: (600, 10)
Supervised test: (600, 28, 28, 1) | Labels: (600, 10)
...
Baseline CNN Test Accuracy: 0.XXXX
Pretrained CNN Test Accuracy: 0.XXXX
```

### Troubleshooting

| Issue | Cause | Fix |
|-------|-------|-----|
| `ModuleNotFoundError: No module named 'tensorflow'` | TensorFlow not installed | `pip install tensorflow` |
| `ValueError: Shapes (None, 28, 28, 1) and (None, 28, 28, 1) are incompatible` | Reshape mismatch | Verify all images are reshaped to `(28, 28, 1)` before feeding to Conv2D layers |
| Autoencoder output looks identical to noisy input | Insufficient training | Increase epochs or check that clean images (not noisy) are used as training targets |
| Transfer model accuracy same as baseline | Encoder weights not transferring | Verify `inputs_izzet` and `e_izzet` reference the same layer objects used in the autoencoder |

---

## 4. Expected Results

| Metric | Baseline CNN | Pretrained CNN |
|--------|-------------|----------------|
| Supervised training samples | 1,800 | 1,800 |
| Supervised validation samples | 600 | 600 |
| Test samples | 600 | 600 |
| Unsupervised pretraining samples | — | 57,000 |
| Test accuracy | ~75–82% | ~78–85% |
| Epochs | 10 | 10 |
| Batch size | 256 | 256 |

**Baseline CNN observations:**
- With only 1,800 training samples, the baseline CNN is data-starved. Training accuracy may climb high but validation accuracy plateaus or diverges, indicating overfitting.
- The confusion matrix shows higher misclassification rates overall compared to Assignment 1 due to the severely limited training set.

**Autoencoder observations:**
- The denoising autoencoder learns to reconstruct clean images from noisy inputs, effectively learning edge and texture features in an unsupervised manner.
- Predicted (denoised) images should show clear structure recovery with some smoothing of fine details.

**Transfer learning observations:**
- The pretrained CNN benefits from encoder weights that already recognize low-level features (edges, textures) learned from 57,000 unlabeled images.
- Validation accuracy should be higher and more stable compared to the baseline, demonstrating that unsupervised pretraining compensates for limited labeled data.
- The accuracy gap between baseline and pretrained models highlights the value of transfer learning in data-scarce scenarios.

---

## 5. Topics Learned

**Unsupervised Learning**
- Autoencoders — neural networks trained to reconstruct their input, learning compressed representations in the bottleneck layer
- Denoising autoencoders — variant trained on corrupted inputs to recover clean outputs, forcing more robust feature learning
- Encoder-decoder architecture — symmetric structure where the encoder compresses input and the decoder reconstructs it

**Transfer Learning**
- Weight transfer — reusing pretrained layer weights in a new model to leverage previously learned features
- Pretraining on unlabeled data — using abundant unlabeled samples to learn general features before fine-tuning on limited labeled data
- Functional API — TensorFlow's flexible model-building interface that supports layer sharing, multiple inputs/outputs, and non-linear topologies

**Advanced Convolution Operations**
- Strided convolutions — using stride > 1 to perform learned downsampling, replacing fixed pooling operations
- Transposed convolutions (Conv2DTranspose) — learnable upsampling that reverses the spatial reduction of strided convolutions
- Same padding — zero-padding input to maintain spatial dimensions before stride application

**Experimental Design**
- Data-scarce simulation — deliberately limiting labeled data to 3,000 samples to create a realistic low-resource scenario
- Controlled comparison — matching all hyperparameters between baseline and pretrained models to isolate the effect of transfer learning
- Noise injection — Gaussian noise with controlled factor as a data augmentation and regularization strategy

---

## 6. Definitions and Key Concepts

| Term | Definition |
|------|-----------|
| **Autoencoder** | A neural network trained to reproduce its input at the output, learning a compressed representation (encoding) in the intermediate layers. |
| **Denoising Autoencoder** | An autoencoder variant trained on corrupted inputs to reconstruct clean originals, forcing the network to learn robust features rather than memorizing identity mappings. |
| **Encoder** | The first half of an autoencoder that compresses input data into a lower-dimensional latent representation. |
| **Decoder** | The second half of an autoencoder that reconstructs the original input from the compressed latent representation. |
| **Transfer Learning** | A technique where knowledge (weights) learned from one task is reused to improve performance on a different but related task. |
| **Pretraining** | Training a model on an auxiliary task (e.g., reconstruction) before fine-tuning it on the target task (e.g., classification), especially useful when labeled data is limited. |
| **Functional API** | TensorFlow/Keras model-building interface that defines models as directed acyclic graphs of layers, enabling layer sharing and complex architectures beyond sequential stacks. |
| **Conv2DTranspose** | A transposed convolutional layer that performs learnable upsampling, reversing the spatial reduction of standard convolutions by inserting zeros between input elements. |
| **Strided Convolution** | A convolution with stride > 1 that simultaneously extracts features and reduces spatial dimensions, replacing separate convolution + pooling operations. |
| **Same Padding** | Zero-padding applied to input so that the output spatial dimensions equal input dimensions divided by stride, preventing information loss at edges. |
| **Mean Squared Error (MSE)** | A loss function computing the average squared difference between predicted and target values, commonly used for reconstruction tasks where pixel-level accuracy matters. |
| **Gaussian Noise** | Random noise sampled from a normal distribution, added to images as a corruption strategy for denoising autoencoder training. |
| **Noise Factor** | A scalar multiplying the noise amplitude before addition to clean images, controlling corruption severity (0.2 in this assignment). |
| **Sigmoid Activation** | An activation function mapping values to the range (0, 1), used on the autoencoder output to match the normalized pixel value range. |
| **Latent Representation** | The compressed feature vector at the autoencoder's bottleneck, capturing the most salient information needed for reconstruction. |
| **Bottleneck Layer** | The narrowest layer in an autoencoder, forcing information compression and serving as the learned feature representation. |
| **Data-Scarce Scenario** | A practical setting where labeled training data is limited and expensive to obtain, motivating techniques like transfer learning and semi-supervised approaches. |
| **Unsupervised Learning** | Learning patterns from data without explicit labels, relying on the structure of the data itself (e.g., reconstruction loss). |
| **Weight Sharing** | Using the same layer instances across multiple models, ensuring transferred weights remain synchronized. |
| **Reconstruction Loss** | The objective function measuring how well an autoencoder reproduces its input, typically MSE or binary crossentropy for pixel values. |

---

## 7. Potential Improvements and Industry Considerations

### Model Architecture

| Current Approach | Industry Alternative | Trade-off |
|-----------------|---------------------|-----------|
| Convolutional autoencoder | Variational Autoencoder (VAE) | Structured latent space enables generation and interpolation, but adds KL-divergence loss complexity |
| Fixed 2-layer encoder | U-Net with skip connections | Better detail preservation in reconstruction via direct encoder-decoder connections, but more memory |
| Conv2DTranspose for upsampling | UpSampling2D + Conv2D | Avoids checkerboard artifacts from transposed convolutions, marginally slower |

### Transfer Learning Strategy

| Current Approach | Industry Alternative | Trade-off |
|-----------------|---------------------|-----------|
| Freeze-free transfer (all weights trainable) | Freeze encoder, train only classifier | Prevents catastrophic forgetting of pretrained features, but limits adaptation to new task |
| Custom autoencoder pretraining | ImageNet-pretrained backbone (ResNet, VGG) | Massive pretrained representations but requires RGB 3-channel input and larger models |
| Single-stage pretraining | Contrastive learning (SimCLR, BYOL) | State-of-the-art self-supervised features, but significantly more complex training pipeline |

### Evaluation and Deployment

| Current Approach | Industry Alternative | Trade-off |
|-----------------|---------------------|-----------|
| Fixed 10 epochs | Early stopping with patience | Automatic convergence detection, especially important with small datasets prone to overfitting |
| Single noise factor (0.2) | Progressive noise scheduling | Curriculum-style training from easy to hard corruption, but requires tuning |
| Matplotlib comparison plots | Weights & Biases experiment tracking | Automated multi-run comparison and hyperparameter sweeps, adds infrastructure dependency |

### Where the Baseline Holds Up

- **The 2-layer convolutional autoencoder** is sufficient for Fashion MNIST's 28×28 resolution. Deeper encoders would compress to sub-pixel spatial dimensions, losing information without meaningful abstraction gain on images this small.
- **Full weight fine-tuning** (no freezing) is appropriate here because the supervised dataset, while small, shares the same domain as the pretraining data. Freezing would be more critical when transferring across domains (e.g., natural images → medical).
- **MSE reconstruction loss** works well for grayscale pixel reconstruction where the output range is bounded [0, 1]. Binary crossentropy would be an alternative but offers minimal practical difference for normalized grayscale images.
