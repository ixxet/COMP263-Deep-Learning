# Assignment 1 — CNNs and RNNs

**Course:** COMP 263 — Deep Learning
**Weight:** Lab Assignment 1 — Due Week 5
**Student:** Izzet Abidi (300898230)

---

## 1. Overview

This assignment applies two fundamentally different deep learning architectures — a **Convolutional Neural Network (CNN)** and a **Recurrent Neural Network (RNN/LSTM)** — to the same image classification task on the **Fashion MNIST** dataset. It sits early in the course progression as the first hands-on lab, establishing a baseline understanding of how architectural inductive biases affect learning on structured visual data. The core competency is building, training, evaluating, and comparing sequential deep learning models end-to-end using TensorFlow/Keras.

---

## 2. Exercise Breakdown

### Exercise 1: Fashion MNIST Classification with CNN and RNN (100 marks)

**Objective:** Build, train, and evaluate both a CNN and an LSTM-based RNN on Fashion MNIST, then compare their classification performance across 10 clothing categories.

**What the script does:**

1. **Data Loading (Step a)** — Imports `fashion_mnist` from TensorFlow and stores training and testing data into two dictionaries (`train_izzet` and `test_izzet`), each with `'images'` and `'labels'` keys.

2. **Initial Exploration (Step b)** — Prints the training set size (60,000 samples), testing set size (10,000 samples), image resolution (28×28 pixels), and the maximum pixel value (255) using `numpy.amax()`.

3. **Data Preprocessing (Step c)** — Normalizes all pixel values to the range [0, 1] by dividing by 255. One-hot encodes labels using `tf.keras.utils.to_categorical()`, converting integer class labels into 10-dimensional binary vectors. Prints the resulting label shapes to confirm the encoding.

4. **Visualization (Step d)** — Defines a reusable `plot_image()` function that displays a single image with its class label, removing axis ticks for a clean presentation. Plots the first 12 training samples in a 4×3 subplot grid with a figure size of 8×8.

5. **Training Data Preparation (Step e)** — Splits the training dataset into 80% training and 20% validation using `sklearn.model_selection.train_test_split()` with `random_state=30` (last two digits of student ID). Stores results in `x_train_izzet`, `x_val_izzet`, `y_train_izzet`, and `y_val_izzet`.

6. **CNN Model (Steps f–g)** — Builds a `Sequential` CNN (`cnn_model_izzet`) with the architecture:
   - Input: (28, 28, 1)
   - Conv2D(32, 3×3, ReLU) → MaxPooling2D(2×2)
   - Conv2D(32, 3×3, ReLU) → MaxPooling2D(2×2)
   - Flatten → Dense(100) → Dense(10, softmax)

   Compiles with Adam optimizer, categorical crossentropy loss, and accuracy metric. Trains for 8 epochs with batch size 256. Evaluates on the test set, generates predictions (`cnn_predictions_izzet`), plots training vs. validation accuracy, visualizes prediction probability distributions for four sample images, and renders a confusion matrix heatmap.

7. **RNN Model (Step h)** — Builds a `Sequential` RNN (`rnn_model_izzet`) with the architecture:
   - Input: (28, 28) — treating each row of 28 pixels as one timestep in a 28-step sequence
   - LSTM(128 units)
   - Dense(10, softmax)

   Repeats the same compilation, training, evaluation, and visualization pipeline as the CNN model, enabling direct architectural comparison.

**Key design decisions:**

- **Image rows as timesteps:** The RNN interprets each 28-pixel row as a single timestep, producing a 28-step sequence per image. This is a natural mapping that preserves spatial structure while fitting the sequential processing paradigm of LSTMs.
- **Batch size 256:** Balances GPU memory efficiency with sufficient gradient diversity per update. Larger batches reduce training noise, which suits the relatively small Fashion MNIST images.
- **Adam optimizer:** Provides adaptive learning rates per parameter, handling the different gradient scales between convolutional and dense layers without manual tuning.
- **No activation on Dense(100):** The intermediate dense layer uses the default linear activation as specified, relying on the preceding ReLU layers for non-linearity and the softmax output for probability calibration.
- **Single channel reshape for CNN only:** The base data remains (28, 28) after normalization. Images are reshaped to (28, 28, 1) only for the CNN path, keeping the LSTM input clean at (28, 28).

**File manifest:**

| File | Purpose |
|------|---------|
| `izzet_linear.py` | Complete implementation — data pipeline, CNN, RNN, all evaluation and visualization |

---

## 3. Runbook

### Prerequisites

```bash
$ pip install tensorflow numpy matplotlib seaborn scikit-learn
```

Requires Python 3.8+ and TensorFlow 2.x. A GPU is beneficial but not required — Fashion MNIST trains within minutes on CPU.

### Execution

```bash
$ cd Assign1/
$ python izzet_linear.py
```

The script runs sequentially: data loading → exploration → preprocessing → visualization → CNN training/evaluation → RNN training/evaluation. All plots display interactively via Matplotlib.

### Sample Output

```
Training dataset size: 60000
Testing dataset size: 10000
Image resolution: (28, 28)
Maximum pixel value: 255
Training labels shape: (60000, 10)
Testing labels shape: (10000, 10)
...
CNN Test Accuracy: 0.XXXX
...
RNN Test Accuracy: 0.XXXX
```

### Troubleshooting

| Issue | Cause | Fix |
|-------|-------|-----|
| `ModuleNotFoundError: No module named 'tensorflow'` | TensorFlow not installed | `pip install tensorflow` |
| `OOM when allocating tensor` | Insufficient GPU memory | Reduce batch size or run on CPU: `CUDA_VISIBLE_DEVICES="" python izzet_linear.py` |
| Plots not displaying | Non-interactive backend | Add `matplotlib.use('TkAgg')` before importing `pyplot`, or run in a GUI-enabled environment |
| `ValueError: Input 0 of layer is incompatible` | Image shape mismatch | Verify reshape to `(28, 28, 1)` for CNN and `(28, 28)` for RNN |

---

## 4. Expected Results

| Metric | CNN | RNN (LSTM) |
|--------|-----|------------|
| Training samples | 48,000 | 48,000 |
| Validation samples | 12,000 | 12,000 |
| Test samples | 10,000 | 10,000 |
| Test accuracy | ~88–91% | ~85–88% |
| Epochs | 8 | 8 |
| Batch size | 256 | 256 |

**CNN observations:**
- The CNN typically converges faster due to parameter sharing through convolutional filters. Training accuracy climbs sharply in the first 2–3 epochs.
- Validation accuracy tracks training accuracy closely, indicating limited overfitting on this dataset with 8 epochs.
- The confusion matrix commonly shows misclassifications between visually similar categories: Shirt ↔ T-shirt/top ↔ Pullover, and Sneaker ↔ Ankle boot.

**RNN observations:**
- The LSTM processes images row-by-row, which discards explicit 2D spatial relationships. This inherently limits its representational advantage compared to convolutions.
- Test accuracy is generally 2–5% lower than CNN, reflecting the architectural mismatch between sequential processing and spatial image data.
- The confusion matrix shows similar misclassification patterns but with slightly higher error rates across all categories.

---

## 5. Topics Learned

**Deep Learning Architectures**
- Convolutional Neural Networks (CNN) — hierarchical spatial feature extraction through learned filter kernels
- Recurrent Neural Networks (RNN) — sequential processing with hidden state propagation across timesteps
- Long Short-Term Memory (LSTM) — gated RNN variant that mitigates vanishing gradients via cell state and forget/input/output gates
- Sequential API — linear stack of layers in Keras for straightforward model construction

**Computer Vision Fundamentals**
- Convolution operation — sliding filter kernel multiplication and summation to detect local patterns
- Max pooling — spatial downsampling that retains dominant features and introduces translation invariance
- Feature maps — intermediate representations produced by convolutional layers at increasing abstraction levels
- Channel dimension — grayscale images require explicit (H, W, 1) reshaping for Conv2D input

**Data Pipeline**
- Pixel normalization — scaling values to [0, 1] for stable gradient computation and faster convergence
- One-hot encoding — converting categorical labels into binary vectors for softmax/crossentropy compatibility
- Train/validation/test splitting — three-way partitioning to tune hyperparameters without contaminating test evaluation

**Model Evaluation**
- Training vs. validation accuracy curves — diagnosing overfitting, underfitting, and convergence behavior
- Confusion matrix — per-class error analysis revealing systematic misclassification patterns
- Prediction probability distribution — per-sample confidence analysis showing softmax output across all classes

---

## 6. Definitions and Key Concepts

| Term | Definition |
|------|-----------|
| **Fashion MNIST** | A dataset of 70,000 grayscale 28×28 images across 10 clothing categories, designed as a drop-in replacement for the original MNIST handwritten digit dataset. |
| **Convolutional Neural Network (CNN)** | A neural network architecture that uses learnable convolutional filters to extract spatial features hierarchically from grid-structured data such as images. |
| **Recurrent Neural Network (RNN)** | A neural network that processes sequential data by maintaining a hidden state that carries information across timesteps. |
| **Long Short-Term Memory (LSTM)** | An RNN variant that uses gating mechanisms (forget, input, output gates) to selectively retain or discard information, mitigating vanishing gradient problems over long sequences. |
| **Convolution** | A mathematical operation that slides a filter kernel across an input, computing element-wise products and summing them to produce a feature map highlighting detected patterns. |
| **Max Pooling** | A downsampling operation that selects the maximum value within a local window, reducing spatial dimensions while retaining the strongest activations. |
| **Flatten** | A layer that reshapes multi-dimensional feature maps into a one-dimensional vector, bridging convolutional layers and fully connected dense layers. |
| **Dense Layer** | A fully connected neural network layer where every input neuron connects to every output neuron through learnable weights. |
| **Softmax** | An activation function that converts a vector of raw scores into a probability distribution where all values sum to 1, used for multi-class classification output. |
| **Categorical Crossentropy** | A loss function that measures the difference between predicted probability distributions and one-hot encoded true labels, penalizing confident incorrect predictions heavily. |
| **Adam Optimizer** | An adaptive learning rate optimization algorithm combining momentum (first moment) and RMSProp (second moment) to adjust step sizes per parameter. |
| **ReLU (Rectified Linear Unit)** | An activation function defined as max(0, x) that introduces non-linearity while avoiding gradient saturation for positive values. |
| **One-Hot Encoding** | A representation scheme that converts integer class labels into binary vectors with a single 1 at the index of the true class and 0s elsewhere. |
| **Normalization** | Scaling input data to a standard range (typically [0, 1]) to ensure consistent gradient magnitudes and faster convergence during training. |
| **Epoch** | One complete pass through the entire training dataset during model training. |
| **Batch Size** | The number of training samples processed together in one forward/backward pass before updating model weights. |
| **Confusion Matrix** | A table showing predicted vs. actual class labels, revealing per-class accuracy and systematic misclassification patterns. |
| **Validation Set** | A held-out portion of training data used to monitor model performance during training without biasing the final test evaluation. |
| **Feature Map** | The output of a convolutional layer, representing the spatial activation pattern detected by a specific filter kernel. |
| **Hidden State** | The internal memory vector in an RNN/LSTM that carries encoded information from previous timesteps to the current computation. |
| **Cell State** | The long-term memory pathway in an LSTM that runs through the entire sequence, regulated by forget and input gates. |
| **Timestep** | A single position in a sequence processed by an RNN. For Fashion MNIST, each row of 28 pixels constitutes one timestep. |

---

## 7. Potential Improvements and Industry Considerations

### Model Architecture

| Current Approach | Industry Alternative | Trade-off |
|-----------------|---------------------|-----------|
| Sequential CNN (2 Conv blocks) | ResNet / EfficientNet with skip connections | Higher accuracy (>95%) but significantly more parameters and training time |
| Single LSTM layer (128 units) | Bidirectional LSTM or Transformer encoder | Better sequence capture from both directions, but doubled parameters and compute |
| Softmax output | Label smoothing + softmax | Regularization against overconfident predictions, marginal accuracy gain |

### Training Strategy

| Current Approach | Industry Alternative | Trade-off |
|-----------------|---------------------|-----------|
| Fixed 8 epochs | Early stopping with patience | Prevents overfitting automatically but requires validation monitoring callback |
| Fixed learning rate (Adam default) | Learning rate scheduling / cosine annealing | Better convergence in later epochs but adds hyperparameter complexity |
| No data augmentation | Random rotations, shifts, flips | Increases effective dataset size and generalization, but adds preprocessing overhead |
| Batch size 256 | Batch size tuning with learning rate scaling | Optimal batch-LR pairing can improve convergence, but requires systematic search |

### Evaluation and Deployment

| Current Approach | Industry Alternative | Trade-off |
|-----------------|---------------------|-----------|
| Matplotlib inline plots | TensorBoard / Weights & Biases | Real-time interactive monitoring across runs, but adds infrastructure dependency |
| Single confusion matrix | Per-class precision, recall, F1-score | More granular performance insight, especially for imbalanced classes |
| Script-based execution | REST API with Flask/FastAPI | Enables real-time inference serving, but adds deployment complexity |

### Where the Baseline Holds Up

- **Fashion MNIST at 28×28 grayscale** is small enough that a shallow CNN reaches near-optimal accuracy without residual connections or attention mechanisms. Adding architectural complexity yields diminishing returns on this particular dataset.
- **8 epochs with batch size 256** provides sufficient convergence for Fashion MNIST. The dataset's simplicity means overfitting typically does not manifest until well beyond 8 epochs, making early stopping unnecessary here.
- **Adam with default learning rate** works well for this task because the loss landscape of Fashion MNIST classification is relatively smooth — there are no sharp minima or pathological gradients that would require careful scheduling.
