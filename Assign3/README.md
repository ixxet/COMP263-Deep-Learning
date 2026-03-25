# Assignment 3 — Variational Autoencoders

**Course:** COMP 263 — Deep Learning
**Weight:** Lab Assignment 3 — Due Week 10
**Student:** Izzet Abidi (300898230)

---

## 1. Overview

This assignment implements a **Variational Autoencoder (VAE)** on Fashion MNIST, introducing probabilistic generative modeling through latent space sampling with the reparameterization trick. It builds on Assignment 2's deterministic autoencoder by replacing the fixed bottleneck with a learned probability distribution, enabling both reconstruction and generation of novel images. The core competency is understanding how KL divergence regularization shapes the latent space into a smooth, continuous manifold that supports meaningful interpolation and sampling.

---

## 2. Exercise Breakdown

### Exercise 1: Variational Autoencoders (100 marks)

**Objective:** Build a complete VAE pipeline — custom sampling layer, convolutional encoder mapping to a 2D latent distribution, convolutional decoder for reconstruction, KL divergence loss, sample generation from the latent grid, and latent space visualization colored by class label.

**What the script does:**

1. **Data Loading (Step a)** — Imports `fashion_mnist` from TensorFlow. Stores the first 60,000 samples in `train_izzet` (images and labels) and the remaining 10,000 in `test_izzet` (images and labels).

2. **Data Preprocessing (Step b)** — Normalizes all pixel values to [0, 1] by dividing by 255. Reshapes images to (28, 28, 1) for Conv2D compatibility. Prints the shapes of `train_izzet['images']` and `test_izzet['images']`.

3. **Custom SampleLayer (Step c.1)** — Implements a custom Keras layer extending `tf.keras.layers.Layer`. The `call()` method takes `[z_mu, z_log_sigma]` as input, computes batch size and dimension via `tf.shape()`, generates random noise from a standard normal distribution using `tf.keras.backend.random_normal()`, and produces latent samples via the reparameterization trick: `z = mu + exp(log_sigma) * epsilon`.

4. **Encoder (Step c.2)** — Builds the encoder using the Functional API (`tf.keras.Model`):
   - Input: (28, 28, 1) → `input_img`
   - Conv2D(32, 3×3, ReLU, same)
   - Conv2D(64, 3×3, ReLU, same, stride 2) — downsamples to 14×14
   - Conv2D(64, 3×3, ReLU, same)
   - Conv2D(64, 3×3, ReLU, same)
   - Flatten → Dense(32, ReLU)
   - `z_mu_izzet`: Dense(2) — latent mean
   - `z_log_sigma_izzet`: Dense(2) — latent log-variance
   - Output: `z_izzet` = SampleLayer()([z_mu, z_log_sigma])

5. **Decoder (Step c.3)** — Builds `decoder_izzet` using the Functional API:
   - Input: (2,) — latent dimension
   - Dense(14 × 14 × 64 = 12,544) — matches encoder's pre-flatten shape
   - Reshape(14, 14, 64)
   - Conv2DTranspose(32, 3×3, ReLU, same, stride 2) — upsamples to 28×28
   - Conv2D(1, 3×3, sigmoid, same) — reconstructed image

6. **VAE Assembly (Step c.4)** — Connects encoder input to decoder output: `vae_izzet = Model(input_img, decoder_izzet(z_izzet))`. Prints the full VAE summary.

7. **Loss and Training (Steps d–e)** — Defines KL divergence loss: `kl_loss = -0.5 * tf.reduce_mean(z_mu - tf.square(z_mu) - tf.exp(z_log_sigma) + 1)`. Adds KL loss via `model.add_loss()`. Compiles with Adam optimizer and MSE reconstruction loss. Trains for 10 epochs at batch size 256 on the training images.

8. **Sample Generation (Step f)** — Uses `tensorflow_probability` to create a 10×10 grid of quantile points from a standard normal distribution (0.05 to 0.95). Decodes each grid point through the decoder and assembles a mosaic of 100 generated Fashion MNIST images.

9. **Latent Space Visualization (Step g)** — Builds a sub-model from the encoder's input to `z_mu_izzet` output. Predicts the latent space encoding of the test set. Plots a 2D scatter plot colored by class label, revealing how the VAE organizes different garment types in the latent space.

**Key design decisions:**

- **Latent dimension = 2:** Chosen specifically for visualization — the entire latent space can be plotted on a 2D scatter. Higher dimensions would improve reconstruction but prevent direct visual inspection of the manifold.
- **Reparameterization trick:** Sampling `z = mu + exp(log_sigma) * epsilon` with deterministic `mu`/`sigma` and stochastic `epsilon` makes the sampling operation differentiable, allowing backpropagation through the stochastic node.
- **Custom Layer class:** Encapsulating the sampling logic in a Keras Layer ensures it integrates cleanly into the computation graph and handles serialization correctly.
- **KL divergence as add_loss:** Using `model.add_loss()` rather than a custom training loop keeps the implementation simple while correctly combining reconstruction (MSE) and regularization (KL) losses.
- **Sigmoid decoder output:** Matches the [0, 1] normalized input range, ensuring MSE loss computes meaningful pixel-level reconstruction error.
- **Strided convolution in encoder, transposed convolution in decoder:** Mirrors Assignment 2's architecture — learned downsampling/upsampling rather than fixed pooling/interpolation.

**File manifest:**

| File | Purpose |
|------|---------|
| `izzet_lab3.py` | Complete implementation — data pipeline, SampleLayer, encoder, decoder, VAE, training, sample generation, latent space visualization |

---

## 3. Runbook

### Prerequisites

```bash
$ pip install tensorflow tensorflow-probability numpy matplotlib
```

Requires Python 3.8+ and TensorFlow 2.x. The `tensorflow-probability` package is needed for the sample generation grid (step f).

### Execution

```bash
$ cd Assign3/
$ python izzet_lab3.py
```

The script runs sequentially: data loading → preprocessing → model building → training → sample generation → latent space plot. Training may take several minutes depending on hardware.

### Sample Output

```
Train images shape: (60000, 28, 28, 1)
Test images shape: (10000, 28, 28, 1)
...
Encoder Summary:
Model: "encoder"
...
Decoder Summary:
Model: "decoder_izzet"
...
VAE Summary:
Model: "vae_izzet"
...
Epoch 1/10
...
Epoch 10/10
...
```

### Troubleshooting

| Issue | Cause | Fix |
|-------|-------|-----|
| `ModuleNotFoundError: No module named 'tensorflow_probability'` | Package not installed | `pip install tensorflow-probability` |
| `ValueError: Dimensions must be equal` in SampleLayer | Shape mismatch between mu and epsilon | Verify `tf.shape()` extracts dimensions from `z_mu` correctly |
| Generated images are blurry/uniform | KL loss dominating, posterior collapse | Check that KL weight is balanced — reconstruction loss should decrease alongside KL |
| Latent scatter shows no class separation | Insufficient training or KL too strong | Train for more epochs or reduce KL weight |

---

## 4. Expected Results

**Training behavior:**
- Total loss decreases over 10 epochs, combining MSE reconstruction loss and KL divergence regularization.
- KL loss increases initially as the encoder learns to use the latent space, then stabilizes as regularization takes effect.

**Sample generation (10×10 grid):**
- The grid shows a smooth transition of garment types across the latent space.
- Images at grid edges may be less coherent (extreme quantile values), while central images are sharper.
- Adjacent grid cells show gradual morphing between garment categories (e.g., shirts blending into coats).

**Latent space visualization:**
- The scatter plot reveals clusters corresponding to Fashion MNIST classes (T-shirts, trousers, sneakers, etc.).
- Similar classes overlap (e.g., shirts and coats) while dissimilar classes separate (e.g., trousers and bags).
- The distribution is approximately Gaussian (centered near origin) due to KL regularization pushing the posterior toward the standard normal prior.
- With latent_dim=2, some class overlap is expected — the bottleneck is intentionally narrow.

---

## 5. Topics Learned

**Variational Inference**
- Variational Autoencoders — generative models that learn a latent probability distribution rather than a fixed encoding, enabling both reconstruction and generation
- KL divergence — measures how much the learned posterior q(z|x) diverges from the prior p(z), regularizing the latent space toward a standard normal distribution
- Evidence Lower Bound (ELBO) — the VAE training objective combining reconstruction likelihood and KL regularization, which the loss function maximizes

**Reparameterization and Sampling**
- Reparameterization trick — reformulating stochastic sampling as a deterministic function of learnable parameters plus external noise, enabling gradient-based optimization through the sampling step
- Custom Keras layers — extending `tf.keras.layers.Layer` to implement novel operations (sampling) that integrate into the TensorFlow computation graph
- Latent space — the compressed representation where each point is a probability distribution parameterized by mean and log-variance, not a fixed vector

**Generative Modeling**
- Decoder as generator — using the trained decoder to map arbitrary latent points to realistic images, enabling generation of samples never seen during training
- Latent space interpolation — traversing the continuous latent manifold to observe smooth transitions between generated outputs
- Prior sampling — drawing latent vectors from the standard normal prior and decoding them to generate novel, diverse samples

**Architecture Patterns**
- Encoder-decoder with stochastic bottleneck — replacing the deterministic bottleneck of a standard autoencoder with a parameterized distribution
- Convolutional VAE — combining spatial feature extraction (Conv2D) with variational inference for image generation
- Loss decomposition — separating the total loss into reconstruction (MSE) and regularization (KL) terms with distinct roles

---

## 6. Definitions and Key Concepts

| Term | Definition |
|------|-----------|
| **Variational Autoencoder (VAE)** | A generative model that learns to encode inputs as probability distributions in latent space and decode samples from those distributions back to data space. |
| **Latent Space** | A low-dimensional representation space where each point corresponds to a potential output; in VAEs, points are sampled from learned distributions. |
| **Latent Dimension** | The number of dimensions in the latent space (2 in this assignment), controlling the capacity and visualizability of the representation. |
| **Reparameterization Trick** | Expressing the random sample z = mu + sigma * epsilon so that gradients flow through mu and sigma while randomness comes from the fixed-distribution epsilon. |
| **KL Divergence** | Kullback-Leibler divergence measures the difference between the learned posterior distribution q(z\|x) and the prior p(z), penalizing deviation from the standard normal. |
| **Evidence Lower Bound (ELBO)** | The VAE optimization objective: maximize reconstruction quality while minimizing KL divergence between the approximate posterior and the prior. |
| **Prior Distribution p(z)** | The assumed distribution over latent variables before observing data — standard normal N(0, I) in this assignment. |
| **Posterior Distribution q(z\|x)** | The encoder's output distribution approximating the true posterior, parameterized by learned mean and log-variance vectors. |
| **z_mu (Mean Vector)** | The encoder output representing the center of the latent distribution for a given input image. |
| **z_log_sigma (Log-Variance)** | The encoder output representing the spread (uncertainty) of the latent distribution; log-space ensures numerical stability and allows negative values. |
| **SampleLayer** | A custom Keras layer implementing the reparameterization trick, producing differentiable samples from the latent distribution. |
| **Reconstruction Loss** | The MSE between the input image and the decoder's output, measuring how well the VAE can reproduce its inputs. |
| **Posterior Collapse** | A failure mode where the encoder ignores input data and outputs a prior-like distribution, making the KL term zero but losing all encoding capability. |
| **Conv2DTranspose** | A transposed convolution performing learnable upsampling, used in the decoder to expand latent representations back to image dimensions. |
| **Functional API** | TensorFlow's model-building interface supporting arbitrary graph topologies — required here for the dual-output encoder (mu and sigma) and layer sharing. |
| **tensorflow_probability** | A library extending TensorFlow with probability distributions and statistical functions, used here for quantile-based latent grid generation. |
| **Quantile Function** | The inverse CDF — maps a probability value to the corresponding point on the distribution, used to create evenly-spaced samples across the latent grid. |
| **Latent Grid** | A regular grid of points in latent space decoded to visualize what the generator has learned across the manifold. |
| **Manifold** | The continuous surface in latent space that the decoder maps to realistic outputs — smooth manifolds enable meaningful interpolation. |
| **Gaussian Encoder** | An encoder that outputs parameters of a Gaussian distribution (mean and variance) rather than a single deterministic vector. |

---

## 7. Potential Improvements and Industry Considerations

### Model Architecture

| Current Approach | Industry Alternative | Trade-off |
|-----------------|---------------------|-----------|
| Latent dim = 2 (for visualization) | Latent dim = 64–256 | Higher capacity captures more detail, but loses direct visualization — requires t-SNE/UMAP for inspection |
| Convolutional VAE | VQ-VAE (Vector Quantized) | Discrete latent codes avoid posterior collapse and produce sharper images, but require codebook management |
| Single-scale encoder | Hierarchical VAE (NVAE, VDVAE) | Multi-scale latent variables capture both global structure and fine detail, at significant training cost |

### Loss and Training

| Current Approach | Industry Alternative | Trade-off |
|-----------------|---------------------|-----------|
| MSE reconstruction loss | Perceptual loss (LPIPS) | Perceptual loss produces sharper, more realistic images by comparing in feature space, but requires a pretrained network |
| Fixed KL weight (1.0) | Beta-VAE (beta > 1) or KL annealing | Beta-VAE encourages disentangled representations; annealing prevents early posterior collapse, but adds hyperparameter tuning |
| 10 epochs, batch 256 | Longer training with learning rate scheduling | More epochs and cosine/warmup scheduling improve final quality, at compute cost |

### Generation and Evaluation

| Current Approach | Industry Alternative | Trade-off |
|-----------------|---------------------|-----------|
| Grid sampling via quantile | Random prior sampling + FID score | FID provides a quantitative generation quality metric, but requires large sample sets and an Inception network |
| 2D scatter for latent space | t-SNE / UMAP of higher-dim latent | Non-linear projections better preserve cluster structure from high-dim spaces, but add a visualization step |
| VAE for generation | Diffusion Models (DDPM) or GANs | State-of-the-art image quality, but orders of magnitude more compute (diffusion) or training instability (GANs) |

### Where the Baseline Holds Up

- **Latent dim = 2** is the correct choice for this assignment because the explicit requirement is to visualize the latent space as a 2D scatter plot and generate a 2D grid of decoded samples. Higher dimensions would break both visualization tasks.
- **MSE reconstruction loss** is appropriate for grayscale 28×28 images where pixel-level fidelity is sufficient. Perceptual loss adds complexity without meaningful benefit at this resolution.
- **The simple KL + MSE formulation** clearly demonstrates the core VAE principle — balancing reconstruction against regularization. More sophisticated losses obscure this pedagogical goal.
