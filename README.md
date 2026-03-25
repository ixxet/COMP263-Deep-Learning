# COMP 263 — Deep Learning

**Student:** Izzet Abidi (300898230)
**Program:** Artificial Intelligence — Software Engineering Technology (AI-SET)
**Institution:** Centennial College — Winter 2026

---

## Repository Overview

This repository contains all graded lab assignments for COMP 263: Deep Learning. Each assignment builds on foundational concepts introduced in earlier weeks, progressing from spatial feature extraction through sequential modeling to advanced architectures.

## Assignment Map

| Assignment | Topic | Core Technique | Status |
|:----------:|-------|----------------|:------:|
| [1](Assign1/) | Fashion MNIST Classification | CNN + RNN (LSTM) | Complete |
| [2](Assign2/) | *Pending* | *TBD* | Upcoming |
| [3](Assign3/) | *Pending* | *TBD* | Upcoming |

## Course Progression

```
Assignment 1                    Assignment 2              Assignment 3
CNN + RNN (LSTM)                TBD                       TBD
───────────────── ────────────► ──────────── ──────────► ────────────
Spatial feature                 ...                       ...
extraction via
convolution layers
    +
Sequential modeling
via LSTM hidden
states on image rows
```

**Assignment 1** establishes the baseline for deep learning classification by comparing two fundamentally different architectures on the same dataset. Convolutional Neural Networks exploit spatial locality through learned filter kernels, while Recurrent Neural Networks (LSTM) process image rows as temporal sequences. Training both on Fashion MNIST reveals how architectural inductive biases shape learning dynamics, convergence speed, and generalization.

## Technology Stack

| Library | Version | Usage |
|---------|---------|-------|
| TensorFlow / Keras | 2.x | Model building, training, evaluation |
| NumPy | 1.x | Array operations, pixel normalization |
| Matplotlib | 3.x | Training curves, image visualization, probability histograms |
| Seaborn | 0.x | Confusion matrix heatmaps |
| scikit-learn | 1.x | Train/validation splitting, confusion matrix computation |

## Quick Start

```bash
# Clone the repository
git clone https://github.com/ixxet/COMP263-Deep-Learning.git
cd COMP263-Deep-Learning

# Install dependencies
pip install tensorflow numpy matplotlib seaborn scikit-learn

# Run Assignment 1
python Assign1/izzet_linear.py
```
