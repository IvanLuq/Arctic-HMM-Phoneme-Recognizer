# CLAUDE.md — Project 2: HMM Phoneme Recognizer

## Project Overview

AI course assignment (Reykjavik University, teacher: Stephan Schiffel).
Build an HMM-based phoneme recognizer using the CMU ARCTIC speech corpus (41 ARPAbet phonemes).
Runs in **Google Colab** with data on **Google Shared Drive**.

## Environment

- **Runtime**: Google Colab (not local)
- **Data on Shared Drive**: `/content/drive/Shareddrives/AI_RU/project2/data/`
- **Local copy** (Windows/WSL): `C:\Users\IVANL\RU\AI\proj2\project2\`
- **Main notebook**: `project2.ipynb`

```python
# Mount drive (run first in every Colab session)
from google.colab import drive
drive.mount('/content/drive')

feature_folder = '/content/drive/Shareddrives/AI_RU/project2/data/_out'
label_folder   = '/content/drive/Shareddrives/AI_RU/project2/data/cmu_us_slt_arctic/lab'
txt_folder     = '/content/drive/Shareddrives/AI_RU/project2/data/txt'
```

## Data Layout

```
data/
  cmu_us_slt_arctic/
    wav/        # .wav audio files (16kHz mono)
    lab/        # .lab phoneme label files (time-stamped ARPAbet)
  _out/         # precomputed .npy feature files (one per utterance)
  txt/
    train.txt   # ~793 utterance basenames (70%)
    val.txt     # ~113 utterance basenames (10%)
    test.txt    # ~226 utterance basenames (20%)
```

Each `.npy` file: shape `(num_frames, 39)` — 13 MFCCs + 13 deltas + 13 delta-deltas, 10ms frames.
Each `.lab` file: rows of `end_time phoneme` (time-stamped ARPAbet).

## What Is Already Implemented ✅

- `pip install python_speech_features`
- Google Drive mount
- File list loading from `train.txt`, `val.txt`, `test.txt`
- `load_labels(label_file, arpabet_dic, n_frames)` — maps `.lab` timestamps to frame labels
- `ARPAbet_dic` — 41-phoneme string→int mapping
- Test feature/label loading (`test_feature_list`, `test_label_list`)
- **π** (initial distribution) — first non-pause phoneme per utterance
- **A** (transition matrix) — bigram counts, Laplace smoothing α=0.001
- **μ, Σ** (emission parameters) — per-phoneme mean/covariance, regularization λ=1e-5
- Transition matrix heatmap

## CRITICAL BUG ⚠️

**All parameter estimation (π, A, μ, Σ) is currently computed from TEST data, not TRAIN data.**

The code loads `train_filenames` but never builds `train_feature_files` / `train_label_files`.
The estimation cells only loop over `test_label_list` / `test_feature_list`.

**Fix**: replicate the file-matching and loading blocks for the train split, then rerun π, A, μ, Σ estimation using `train_feature_list` and `train_label_list`.

## What Needs To Be Implemented ❌

### 1. Fix train split loading (prerequisite for everything else)
```python
train_feature_files = []
train_label_files = []
for feature_idx, feature_filename in enumerate(feature_files):
    feature_basename = Path(feature_filename).stem
    for train_filename in train_filenames:
        if feature_basename == train_filename:
            train_feature_files.append(feature_filename)
            train_label_files.append(label_files[feature_idx])
```

### 2. Forward Algorithm (Filtering)
- Function stub `filtering()` exists but is empty
- Compute `α_i(t) = P(O_{1:t}, X_t = i)` for each frame
- Use **log-space or scaling** to avoid underflow (multivariate Gaussian over 39 dims goes to zero fast)
- Return `P(X_t | O_{1:t})` — normalize α at each step

### 3. Viterbi Algorithm
- Compute most likely state sequence: `argmax P(X_{1:T} | O_{1:T})`
- Use **log-probabilities** throughout
- Need backpointer matrix for traceback
- Return predicted label sequence of same length as input

### 4. Evaluation
- Frame accuracy for Filtering and Viterbi on both train and test sets
- Confusion matrix (test set)
- Per-phoneme precision and recall (test set)
- Example utterances: predicted vs ground truth phoneme sequences

## Gaussian Emission Probability

```python
from scipy.stats import multivariate_normal

def emission_prob(obs, mean, cov):
    return multivariate_normal.logpdf(obs, mean=mean, cov=cov)
```

Use `logpdf` to stay in log-space. The 39-dim Gaussian has very small probabilities in linear space.

## Grading Breakdown

| Component | Weight | Status |
|---|---|---|
| Data preparation | 15% | ✅ (fix train split bug) |
| Supervised parameter estimation | 15% | ✅ (fix train split bug) |
| Filtering + Viterbi | 20% | ❌ not done |
| Evaluation & plotting | 15% | ❌ not done |
| Analysis & discussion | 10% | ❌ not done |
| Report (4–6 pages PDF) | 15% | ❌ not done |
| Code quality | 10% | partial |
| Optional extensions | +20 bonus | not started |

## HMM Parameter Summary

| Symbol | Shape | Description |
|---|---|---|
| `initial_distribution` | (41,) | π — P(X_1 = i) |
| `transition_matrix` | (41, 41) | A — P(X_{t+1}=j \| X_t=i), rows sum to 1 |
| `emission_parameters[i][0]` | (39,) | μ_i — mean for phoneme i |
| `emission_parameters[i][1]` | (39, 39) | Σ_i — covariance for phoneme i |

## Key Constants

```python
n_phonemes = 41
frame_step = 0.01        # 10ms
alpha = 0.001            # Laplace smoothing for transition matrix
lam = 0.00001            # Tikhonov regularization for covariance
```

## Numerical Stability Notes

- Always use **log-probabilities** in Filtering and Viterbi — linear probs underflow to 0 within a few frames
- For Filtering: use scaled α (divide by sum at each step) OR work in log-space with log-sum-exp
- For Viterbi: log-space is straightforward (additions instead of multiplications)
- Covariance regularization (+ λI) prevents singular matrices in multivariate Gaussian
