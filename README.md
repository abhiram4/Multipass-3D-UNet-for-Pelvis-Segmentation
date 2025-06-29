# 3D MultiPass U‑Net: Enhanced Spatial Context Pelvic CT Segmentation

A specialized 3D U‑Net architecture leveraging a **MultiPass Learning** strategy to propagate spatial context across multiple inference passes, leading to improved segmentation accuracy on pelvic CT scans.

---

## 🔍 Overview

Pelvic CT segmentation is challenging due to complex anatomy and large volumetric data. Our **3D MultiPass U‑Net** addresses these by performing **dual‑inference passes**, where coarse and fine contextual information is iteratively exchanged to refine predictions.

Key advantages:

* Captures both local details and global context
* Reduces false positives in ambiguous regions
* Maintains volumetric consistency through 3D convolutions

---

## 🚀 Key Features&#x20;

### 1. MultiPass Learning

* **Dual-Inference Mechanism**: First pass produces a coarse segmentation map. Second pass refines this map by incorporating spatial context from the initial inference.
* **Spatial Context Propagation**: Coarse predictions are concatenated with original CT volumes, enabling the network to focus on challenging regions in the subsequent pass.
* **Iterative Refinement**: Improves boundary delineation and reduces misclassifications by learning residual corrections.

### 2. 3D U‑Net Backbone

* Encoder‑Decoder structure with skip connections for combining low‑level and high‑level features.
* 3D convolutions preserve volumetric information.

### 3. Medical Imaging Pipeline

* Automatic resampling, intensity normalization, and cropping to standardize inputs.

### 4. Advanced Training Strategies

* **Mixed‑Precision Training**: Balanced speed and memory usage using NVIDIA AMP.
* **Gradient Clipping**: Prevents exploding gradients for stable convergence.
* **Learning Rate Scheduling**: Cosine annealing or plateau reduction for optimal learning.
* **MONAI 3D Augmentations**: Random affine, elastic deformations, flips, and intensity shifts in 3D.

---

## 📊 Dataset Information&#x20;

### CTPelvic1K Dataset

A curated collection of pelvic CT scans with expert-annotated segmentations.

**Download**: [AIDA Data Hub](https://datahub.aida.scilifelab.se/10.23698/aida/ctpel)

**Directory Structure**:

```bash
ctpelvic1k/
└── Patient_XX/
    ├── im_1/       # CT DICOM series (e.g., axial slices)
    └── im_3/       # Segmentation DICOM (label maps)
```

---

## ⚙️ Installation&#x20;

```bash
# Clone repository
git clone https://github.com/DigiDxDoc/CT-Segmentation.git
cd CT-Segmentation

# Create a virtual environment using Python 3.10
python3.10 -m venv venv

# Activate the virtual environment
# On macOS/Linux:
source venv/bin/activate
# On Windows:
venv\Scripts\activate

# Install dependencies from requirements.txt
pip install -r requirements.txt
```

> **Note**: Ensure you have the DCMQI toolkit installed and added to your PATH for the `segimage2itkimage` command used in processing. See [https://github.com/QIICR/dcmqi](https://github.com/QIICR/dcmqi) for installation instructions. Python 3.10 should be used because MONAI doesn't support any further versions.

---

## 🏃 Usage&#x20;

1. **Prepare Dataset and Directories**:

   * Download the CTPelvic1K dataset from AIDA Data Hub.
   * Create a directory named `ctpel` in your project root and place patient folders (`Patient_XX`) inside, preserving the `im_1` and `im_3` subfolders.
   * Create a directory named `processed_files` in the project root to store processed outputs.

   Your project structure should look like this:

   ```bash
   CT-Segmentation/
   ├── ctpel/
   │   ├── Patient_01/
   │   │   ├── im_1/
   │   │   └── im_3/
   │   ├── Patient_02/
   │   │   ├── im_1/
   │   │   └── im_3/
   │   └── ...
   ├── processed_files/  # Directory for processed data
   ├── processing.py
   ├── training.py
   ├── README.md
   └── venv/
   ```

2. **Process the dataset**:

   ```bash
   python processing.py
   ```

   * Converts DICOM CT and SEG to standardized volumes (128×128×72).
   * Saves individual patient `.pkl` files and a `complete_dataset.pkl` in `processed_files/`.

3. **Train the model**:

   ```bash
   python training.py
   ```

   * Loads processed data from `processed_files/complete_dataset.pkl`.
   * Trains the MultiPass U‑Net and checkpoints the best model as `best_model.pth`.

> The training script auto-detects CUDA; falls back to CPU if unavailable.

---

## 💻 Inference Pipeline&#x20;

To run inference with your trained model, follow these steps:

### 📦 Setup

```bash
# Activate your Python 3.10 virtual environment
python3.10 -m venv venv
# On macOS/Linux:
source venv/bin/activate
# On Windows:
venv\Scripts\activate

# Install inference dependencies
pip install -r requirements.txt

```

### 🔧 File Structure

Ensure your repository includes the following for inference:

```bash
inference_pipeline/                   # Inference pipeline folder
├── processing.py                     # Preprocessing and loading functions
├── run_inference.py                  # Script to execute model inference
├── best_model.pth                    # Pretrained model checkpoint
├── test_data/                        # Sample test volumes for inference
└── processed_files/                  # Directory to save inference outputs
```

* `processing_pipeline.py` handles loading, resampling, and normalization of input volumes.
* `run_inference.py` loads `best_model.pth`, runs predictions, and saves segmentation masks to `processed_files/inference_results`.

---

## 📊 Training Progress

During training, you will see:

* A batch-level progress bar with loss and dice score metrics.
* Epoch summaries showing average loss and dice score.
* Notifications when `best_model.pth` is updated with a new best dice score.

---
