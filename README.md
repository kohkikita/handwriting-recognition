## Handwriting Recognition (EMNIST, Streamlit + Multiple Models)

This project is a **handwriting recognition** system built on the **EMNIST** dataset with:
- **CNN (PyTorch, MPS)** – main high‑accuracy model
- **SVM, K‑Nearest Neighbors, HistGradientBoosting (scikit‑learn)** – fast classical baselines
- A **Streamlit** web app to upload an image and see the recognized text.

The goal is to reliably read text like **“HELLO 12345 WORLD”** from simple handwritten images, optimized for an **M2 Pro MacBook**.

---

## Project Structure

- **`src/app/streamlit_app.py`** – Streamlit UI (model selector, image upload, predictions).
- **`src/data/preprocessing.py`** – image preprocessing & character segmentation.
- **`src/data/dataset_utils.py`** – EMNIST loading and **orientation fix at source**.
- **`src/models/`**
  - `cnn_model.py` – `SimpleCNN` (with batch norm & dropout).
  - `knearest_model.py` – KNN wrapper.
  - `svm_mode.py` – Linear SVM builder.
  - `gradient_model.py` – Gradient model placeholder / loader.
- **`src/training/`**
  - `train_cnn.py`, `train_svm.py`, `train_knearest.py`, `train_gradient.py` – per‑model scripts.
- **`train_all_models.py`** – main script to train **all** models end‑to‑end.
- **`data/raw`** – EMNIST data (downloaded automatically).
- **`data/processed`** – any processed/derived data.

---

## 1. Environment Setup

From the project root:

```bash
python3 -m venv .venv
source .venv/bin/activate  # macOS / Linux
pip install --upgrade pip
pip install -r requirements.txt
```

Make sure **PyTorch with Metal (MPS)** support is installed for best CNN speed on M2:

```bash
python -c "import torch; print(torch.backends.mps.is_available())"
```

This should print `True` on an M2 Pro with the right PyTorch build.

---

## 2. Training All Models

To train **all** models (SVM, KNN, HistGradientBoosting, CNN) with the optimized settings:

```bash
python3 train_all_models.py
```

What this script does:
- Downloads **EMNIST** if needed.
- Applies the **orientation fix at source** so characters are upright:
  - Flip horizontally.
  - Rotate **90° counter‑clockwise**.
- Flattens images and applies **PCA (784 → 100)** for sklearn models.
- Trains:
  - **LinearSVC** (C and max_iter tuned for speed/accuracy).
  - **KNearestModel** (n_neighbors tuned, `n_jobs=-1` for parallel CPU).
  - **HistGradientBoostingClassifier** (multithreaded, tuned depth/iter).
  - **SimpleCNN** with **MPS GPU**, 20 epochs + LR scheduler.
- Saves models and PCA transformers in `models/`:
  - `svm_emnist.joblib`, `svm_pca.pkl`
  - `knn_model.pkl`, `knn_pca.pkl`
  - `gradient_model.pkl`, `gradient_pca.pkl`
  - `cnn_mnist.pth`

To train individual models instead, use the scripts in `src/training/`:

```bash
python3 -m src.training.train_cnn
python3 -m src.training.train_svm
python3 -m src.training.train_knearest
python3 -m src.training.train_gradient
```

---

## 3. Running the Streamlit App

From the project root:

```bash
streamlit run src/app/streamlit_app.py
```

The app lets you:
- Choose the **model**: `CNN`, `SVM`, `KNearest`, `GradientBoosting`.
- **Upload** an image containing characters (e.g., “HELLO 12345 WORLD”).
- See:
  - The **recognized text** in a textbox.
  - Per‑character predictions and confidences.

The app:
- Segments the image into **individual characters** (top‑to‑bottom, left‑to‑right).
- Resizes each character to **28×28**, normalizes to `[0, 1]`, and:
  - For sklearn models: applies **PCA → classifier**.
  - For CNN: feeds the 28×28 tensor directly.

> **Tip:** Use the **CNN** model for best accuracy on EMNIST‑style characters.

---

## 4. EMNIST Orientation Fix (Important)

Raw **EMNIST** images are **rotated 90° counter‑clockwise and flipped horizontally**.

This project fixes orientation **inside `dataset_utils.py`**:
- `_emnist_fix_orientation(...)`:
  - Flips horizontally.
  - Rotates **90° counter‑clockwise**.
- `get_emnist_datasets(apply_orientation_fix=True)` is used in `train_all_models.py`.

Because the dataset is fixed at the source:
- **Training** sees upright characters.
- **Inference** (Streamlit) expects **upright user images** with **no extra rotation/flip**.

If your uploaded images appear rotated in the UI:
- Adjust your drawing / image rotation so characters are upright before upload.

---

## 5. Models Overview

- **CNN (`SimpleCNN`)**
  - 2 conv layers + batch norm + max‑pool + dropout + 2 FC layers.
  - Trained with Adam, LR scheduler, 20 epochs, MPS acceleration.
  - Best overall accuracy (~88%+ on EMNIST test).

- **LinearSVM**
  - Trained on **PCA‑reduced** features (100 dims).
  - Good speed / strong linear baseline.

- **KNN**
  - `n_neighbors` tuned, `weights='distance'`, `n_jobs=-1`.
  - Very fast training, decent accuracy.

- **HistGradientBoosting**
  - Tree‑based model with multithreading.
  - Good accuracy; slower to train, fast to infer.

---

## 6. Troubleshooting

- **Models not found / “Model not found” in Streamlit**
  - Make sure you have run:
    ```bash
    python3 train_all_models.py
    ```
  - Confirm the `models/` directory contains the `.pth`, `.joblib`, and `.pkl` files.

- **“ModuleNotFoundError: No module named 'src'”**
  - Run commands from the **project root**:
    ```bash
    cd /Users/kohki/Coding/handwriting-recognition
    ```
  - `streamlit_app.py` and `train_all_models.py` already add the project root to `sys.path`.

- **GPU (MPS) not used**
  - Check:
    ```bash
    python -c "import torch; print(torch.backends.mps.is_available())"
    ```
  - If `False`, install a PyTorch build with Metal support and re‑run training.

- **Predictions still wrong for your custom image**
  - Ensure:
    - Characters are **dark on light background** (or similar to EMNIST).
    - Characters are upright and not rotated.
    - There is clear spacing between characters so segmentation can separate them.

---

## 7. Extending the Project

- Add new architectures (e.g., deeper CNNs, transformers) in `src/models/`.
- Tune hyperparameters in `train_all_models.py` (PCA size, SVM C, KNN neighbors, CNN epochs).
- Improve segmentation heuristics in `src/data/preprocessing.py` for more complex layouts.

This setup is designed to be a **fast, usable baseline** for experimenting with handwritten character recognition. Feel free to adapt it to your own datasets and UI. 
