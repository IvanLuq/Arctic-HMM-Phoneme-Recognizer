# HMM Phoneme Recognizer Project

## Overview

This project implements a **Hidden Markov Model (HMM)** for automatic speech phoneme recognition using aligned **MFCC** feature data and phoneme labels.

The system models speech as a sequence of hidden phoneme states and supports three inference methods:

- **Forward Filtering**
- **Viterbi Decoding**
- **Forward-Backward Smoothing** *(added as an extension to the original project)*

The project includes:

- Supervised estimation of HMM parameters from labelled training data
- Frame-level phoneme prediction
- Evaluation on training, validation, and test sets
- Confusion matrices and per-phoneme precision/recall analysis
- Example utterance comparisons between predicted and true phoneme sequences

The phoneme inventory is based on the **41-symbol ARPAbet set**, including vowels, consonants, and silence/pause. Each observation frame is represented as a **39-dimensional MFCC feature vector**.

---

## Model Description

### Hidden State Space

The hidden state variable represents the spoken phoneme at each frame.  
Each state corresponds to one of the 41 phoneme symbols in the dataset.

### Observation Model

Each frame is represented by a feature vector:

- `O_t ∈ R^39`

The emission distribution for each phoneme is modeled as a **multivariate Gaussian** with:

- Mean vector `μ_i`
- Covariance matrix `Σ_i`

---

## HMM Parameters

The model is defined by the following components:

- **Initial state distribution** `π`
- **Transition matrix** `A`
- **Emission means** `μ`
- **Emission covariance matrices** `Σ`

These parameters are estimated in a supervised way from aligned training data.

---

## Parameter Estimation

### 1. Initial State Distribution
The initial distribution is estimated by counting how often each phoneme appears as the first state in an utterance.

### 2. Transition Matrix
The transition matrix is estimated by counting adjacent phoneme transitions across the training set.  
Additive smoothing is applied to avoid zero-probability transitions.

### 3. Emission Parameters
For each phoneme, all corresponding MFCC frames are collected from the training data and used to compute:

- The sample mean vector
- The sample covariance matrix

A small regularization term is added to covariance matrices to improve numerical stability. :contentReference[oaicite:2]{index=2}

---

## Inference Methods

### Forward Filtering
Computes the posterior belief over phoneme states at each frame using only past and current observations.

### Viterbi Decoding
Finds the single most likely phoneme sequence using dynamic programming in log-space.

### Forward-Backward Smoothing
Computes the posterior state probabilities using both past and future observations.  
This was implemented as an additional inference method to compare against Filtering and Viterbi. :contentReference[oaicite:3]{index=3}

---

## Numerical Stability

To make the implementation robust:

- Forward probabilities are normalized at each timestep
- Viterbi is computed fully in **log-space**
- Emission probabilities are clamped before taking logs
- Covariance matrices are regularized to avoid singularities

---

## Evaluation

The model is evaluated using:

- **Frame-level accuracy**
- **Confusion matrices**
- **Per-phoneme precision, recall, and F1-score**
- **Example utterance comparisons**

According to our results, **Viterbi performed slightly better than Filtering and Smoothing**, achieving the best validation performance among the three methods. :contentReference[oaicite:4]{index=4}

---

## Prerequisites

- Python **3.8 or higher**

Required packages:

- `numpy`
- `scipy`
- `matplotlib`

Install all dependencies with:

```bash
pip install numpy scipy matplotlib
