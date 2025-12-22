import streamlit as st
import numpy as np
import cv2
import torch
import os
import sys
import joblib
from pathlib import Path

# Add project root to path for imports
project_root = Path(__file__).parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.models.cnn_model import SimpleCNN
from src.models.knearest_model import KNearestModel
from src.models.svm_mode import build_svm
from src.data.preprocessing import preprocess_image, segment_characters
from src.config import NUM_CLASSES, LABELS_FALLBACK

def label_to_char(label):
    """Convert numeric label to character"""
    if 0 <= label < len(LABELS_FALLBACK):
        return LABELS_FALLBACK[label]
    return str(label)

st.title("Handwriting Recognition Demo")
st.caption("Recognizes digits (0-9) and uppercase letters (A-Z) - 47 classes total")

# Model selection
model_type = st.selectbox(
    "Choose Model Method",
    ["CNN", "SVM", "KNearest", "GradientBoosting"],
    help="CNN uses Metal GPU, others use optimized parallel processing with PCA"
)

# Model paths (relative to project root)
_model_dir = project_root / "models"
MODEL_PATHS = {
    "CNN": str(_model_dir / "cnn_mnist.pth"),
    "SVM": str(_model_dir / "svm_emnist.joblib"),
    "KNearest": str(_model_dir / "knn_model.pkl"),
    "GradientBoosting": str(_model_dir / "gradient_model.pkl")
}

@st.cache_resource
def load_cnn_model(model_path):
    """Load CNN model"""
    if not os.path.exists(model_path):
        return None
    model = SimpleCNN(num_classes=NUM_CLASSES)
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()
    return model

@st.cache_resource
def load_sklearn_model(model_path, model_type):
    """Load sklearn model (SVM, KNN, etc.) with PCA"""
    if not os.path.exists(model_path):
        return None, None
    
    # Load PCA transformer - check multiple possible paths
    pca = None
    pca_paths = [
        model_path.replace(".joblib", "_pca.pkl").replace(".pkl", "_pca.pkl"),
        f"models/{model_type.lower()}_pca.pkl",
        "models/pca_transformer.pkl"  # Fallback to general PCA
    ]
    
    for pca_path in pca_paths:
        full_pca_path = project_root / pca_path if not os.path.isabs(pca_path) else Path(pca_path)
        if full_pca_path.exists():
            try:
                pca = joblib.load(str(full_pca_path))
                break
            except Exception as e:
                continue
    
    if pca is None:
        # Try one more time with absolute path from models directory
        models_dir = project_root / "models"
        for pca_file in ["svm_pca.pkl", "knn_pca.pkl", "gradient_pca.pkl", "pca_transformer.pkl"]:
            pca_file_path = models_dir / pca_file
            if pca_file_path.exists():
                try:
                    pca = joblib.load(str(pca_file_path))
                    break
                except Exception:
                    continue
    
    if pca is None:
        st.error(f"❌ PCA transformer not found for {model_type}. Please retrain models using train_all_models.py")
    
    if model_type == "KNearest":
        # KNearestModel has a custom load method
        from src.models.knearest_model import KNearestModel
        model = KNearestModel.load(model_path)
        return model, pca
    else:
        # Standard sklearn models
        model = joblib.load(model_path)
        return model, pca

def predict_cnn(model, processed_img):
    """Predict using CNN"""
    tensor = torch.from_numpy(processed_img).unsqueeze(0).unsqueeze(0).float()
    with torch.no_grad():
        logits = model(tensor)
        probs = torch.softmax(logits, dim=1).numpy()[0]
    pred_label = int(np.argmax(probs))
    confidence = float(probs[pred_label])
    return pred_label, confidence, probs

def predict_sklearn(model, processed_img, pca=None):
    """Predict using sklearn models with optional PCA"""
    # Flatten image for sklearn models (28x28 -> 784)
    # Ensure it's the right shape and type
    if processed_img.shape != (28, 28):
        raise ValueError(f"Expected image shape (28, 28), got {processed_img.shape}")
    
    flattened = processed_img.flatten().reshape(1, -1).astype(np.float32)
    
    # Verify shape is (1, 784)
    if flattened.shape != (1, 784):
        raise ValueError(f"After flattening, expected shape (1, 784), got {flattened.shape}")
    
    # CRITICAL: Apply PCA if available (models were trained on PCA-transformed data)
    if pca is not None:
        try:
            flattened = pca.transform(flattened)
            # After PCA, should be (1, 100)
            if flattened.shape[1] != 100:
                raise ValueError(f"After PCA, expected 100 features, got {flattened.shape[1]}")
        except Exception as e:
            raise ValueError(f"PCA transformation failed: {e}. Input shape: {flattened.shape}, Expected: (1, 784)")
    else:
        # If PCA is None but model expects PCA, this will fail
        # Models trained with PCA require PCA transformation
        raise ValueError("PCA transformer is required but not loaded. Please retrain models using train_all_models.py")
    
    # Handle KNearestModel wrapper
    if hasattr(model, 'model'):
        # It's a KNearestModel wrapper
        pred = model.predict(flattened)[0]
        if hasattr(model.model, 'predict_proba'):
            probs = model.model.predict_proba(flattened)[0]
            confidence = float(probs[pred])
        else:
            probs = None
            confidence = 1.0
    else:
        # Standard sklearn model
        pred = model.predict(flattened)[0]
        if hasattr(model, 'predict_proba'):
            probs = model.predict_proba(flattened)[0]
            confidence = float(probs[pred])
        else:
            probs = None
            confidence = 1.0
    
    return int(pred), confidence, probs

def apply_emnist_orientation_fix(img_array):
    """
    Apply orientation fix to match EMNIST training preprocessing.
    EMNIST training uses: torch.rot90(img, k=1, dims=[1,2]).flip(2)
    This is: rotate 90° counter-clockwise, then flip horizontally
    img_array: numpy array of shape (H, W) or (28, 28)
    """
    # Match PyTorch transformation exactly: rotate 90° counter-clockwise, then flip horizontally
    # This matches what the models were trained with
    rotated = cv2.rotate(img_array, cv2.ROTATE_90_COUNTERCLOCKWISE)
    flipped = cv2.flip(rotated, 1)  # 1 = horizontal flip
    
    return flipped

def recognize_text_from_image(img, model, model_type):
    """Segment image into characters and recognize each one"""
    # Segment characters from image
    chars, processed_img = segment_characters(img)
    
    if len(chars) == 0:
        return "", "No characters detected in image.", []
    
    recognized_text = ""
    details = []
    char_images = []  # For visualization
    
    # Process each character
    for i, (char_img, bbox) in enumerate(chars):
        # Don't apply EMNIST orientation fix - user's images are already in normal orientation
        # The EMNIST fix was needed for EMNIST dataset, but user's handwritten images
        # are already correctly oriented, so we use them as-is
        # Normalize character image (it's already 28x28 from segment_characters)
        # Convert to float and normalize to [0, 1] range (matching EMNIST preprocessing)
        # EMNIST uses ToTensor() which normalizes to [0,1], so we match that
        char_normalized = char_img.astype(np.float32) / 255.0
        
        # Ensure shape is correct (28, 28)
        if char_normalized.shape != (28, 28):
            # Resize if needed
            char_normalized = cv2.resize(char_normalized, (28, 28), interpolation=cv2.INTER_AREA)
            char_normalized = char_normalized.astype(np.float32) / 255.0 if char_normalized.max() > 1.0 else char_normalized.astype(np.float32)
        
        # Predict character
        if model_type == "CNN":
            pred_label, confidence, probs = predict_cnn(model, char_normalized)
        else:
            pred_label, confidence, probs = predict_sklearn(model, char_normalized, pca)
        
        char = label_to_char(pred_label)
        recognized_text += char
        details.append(f"Char {i+1}: '{char}' (label {pred_label}, confidence: {confidence:.2%})")
        # Show original character (before orientation fix) so user can see normal orientation
        char_images.append((char_img, char, confidence))
    
    return recognized_text, "\n".join(details), char_images

# File uploader
uploaded_file = st.file_uploader("Upload a handwritten image", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    # Read and display image
    file_bytes = np.frombuffer(uploaded_file.read(), np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_GRAYSCALE)
    
    if img is None:
        st.error("Could not decode image. Please upload a valid image file.")
    else:
        st.image(img, caption="Uploaded Image", use_container_width=True)
        
        # Load model
        model_path = MODEL_PATHS[model_type]
        model = None
        
        if model_type == "CNN":
            model = load_cnn_model(model_path)
            pca = None
        else:
            model, pca = load_sklearn_model(model_path, model_type)
        
        if model is None:
            st.warning(f"Model not found at {model_path}. Please train the model first.")
        else:
            # Recognize text from image
            try:
                with st.spinner("Processing image and recognizing characters..."):
                    recognized_text, details, char_images = recognize_text_from_image(img, model, model_type)
                
                # Display recognized text in textbox
                st.subheader("Recognized Text")
                st.text_area(
                    "Text",
                    recognized_text,
                    height=100,
                    key="recognized_text",
                    help="The recognized text from the image"
                )
                
                # Show character previews
                if char_images:
                    st.subheader("Detected Characters")
                    cols = st.columns(min(5, len(char_images)))
                    for idx, (char_img, char, conf) in enumerate(char_images):
                        with cols[idx % len(cols)]:
                            st.image(char_img, caption=f"'{char}' ({conf:.0%})", use_container_width=True)
                
                # Show details in expander
                with st.expander("Recognition Details"):
                    st.text(details)
                    
            except Exception as e:
                st.error(f"Error processing image: {str(e)}")
                import traceback
                st.code(traceback.format_exc())
