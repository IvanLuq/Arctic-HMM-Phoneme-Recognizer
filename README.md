================================================================================
  Project 2: HMM Phoneme Recognizer
  README
================================================================================

OVERVIEW
--------
This project implements a Hidden Markov Model (HMM) for automatic speech
phoneme recognition. It covers:

  - Supervised HMM parameter estimation (initial distribution, transition
    matrix, Gaussian emission models) using aligned MFCC training data.
  - Three inference algorithms: Forward Filtering, Viterbi Decoding, and
    Forward-Backward Smoothing which was an addition we decided to make.
  - Comprehensive evaluation with frame-level accuracy, confusion matrices,
    and per-phoneme precision/recall

The phoneme inventory is the 41-symbol ARPAbet set (vowels, consonants,
and silence/pause).


PREREQUISITES
-------------
Python version: 3.8 or higher

Required packages:
  numpy
  scipy
  matplotlib

Install all dependencies at once with:

    pip install numpy scipy matplotlib


DATA REQUIREMENTS
-----------------
The notebook expects the following directory structure relative to the
notebook file (it is prepared to work both on google colab and locally on VS code):

  data/
  ├── _out/# Pre-extracted MFCC feature files (.npy)
  │     └── <utterance_id>.npy
  ├── cmu_us_slt_arctic/
  │     └── lab/ # Phoneme label files (.lab)
  │           └── <utterance_id>.lab
  └── txt/ # Train/val/test split lists
        ├── train.txt
        ├── val.txt
        └── test.txt

Feature files (.npy): NumPy arrays of shape (n_frames, n_features)
  that contain pre-computed MFCC coefficients at a 10 ms frame step.

Label files (.lab): Space-delimited text files with one phoneme boundary
  per line in the format:  <end_time>  <field>  <phoneme_label>

Split files (.txt): One utterance filename stem per line, defining which
  utterances belong to the training, validation, and test sets.


HOW TO RUN
----------

Option 1 — VS Code
  1. Open the folder containing `project2.ipynb` in VS Code.
  2. Install the "Jupyter" extension if not already installed.
  3. Open `project2.ipynb` and click "Run All" at the top of the notebook,
     or run cells individually with Shift+Enter.
  4. Select a Python kernel that has all required packages installed.

Option 2 — Google Colab
  1. Upload `project2.ipynb` to Google Colab (or open it from Google Drive).
  2. Mount your Google Drive when prompted (the notebook contains a Drive
     mount cell); ensure the `data/` directory is placed at:
         /content/drive/Shareddrives/AI_RU/project2/data
  3. Run all cells in order.


EXPECTED OUTPUTS
----------------
After running all cells the notebook produces:

  - Trained HMM parameters:
      * Initial state distribution (π), shape (41,)
      * Transition matrix (A), shape (41, 41)
      * Gaussian emission parameters (mean μ and covariance Σ) per phoneme

  - Frame-level accuracy (%) on train, validation, and test sets for each
    of the three inference methods (Filtering, Viterbi, Smoothing)

  - Confusion matrix heatmaps (41×41) and per-phoneme metrics table
    (precision, recall, F1) for each inference method

  - Example utterance visualizations comparing predicted phoneme sequences
    against ground-truth labels


TROUBLESHOOTING
---------------
- "Module not found" errors: Verify all packages listed under Prerequisites
  are installed in the active Python environment/kernel.

- Data path errors: Confirm that the `data/` folder exists alongside the
  notebook and contains the three required subdirectories (_out/, lab/, txt/).

- Memory issues on full evaluation: Reduce `max_eval_utterances` to limit
  the number of utterances processed during evaluation.

- Slow inference: Viterbi and Forward-Backward smoothing are O(T * K^2)
  per utterance; expect several minutes for the full test set on a CPU.

================================================================================
