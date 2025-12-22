# src/data/preprocessing.py

import cv2
import numpy as np
from src.config import IMG_SIZE

def to_grayscale(img_bgr_or_gray: np.ndarray) -> np.ndarray:
    if len(img_bgr_or_gray.shape) == 3:
        return cv2.cvtColor(img_bgr_or_gray, cv2.COLOR_BGR2GRAY)
    return img_bgr_or_gray

def binarize(gray: np.ndarray) -> np.ndarray:
    # Otsu threshold; invert so text is white on black (common for MNIST/EMNIST style)
    _, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    return th

def resize_and_center(char_img: np.ndarray, size: int = IMG_SIZE) -> np.ndarray:
    """
    char_img is a binary image (white char on black background).
    We crop tight, resize preserving aspect ratio, then pad to size x size.
    """
    ys, xs = np.where(char_img > 0)
    if len(xs) == 0 or len(ys) == 0:
        return np.zeros((size, size), dtype=np.uint8)

    x0, x1 = xs.min(), xs.max()
    y0, y1 = ys.min(), ys.max()
    crop = char_img[y0:y1+1, x0:x1+1]

    h, w = crop.shape
    if h == 0 or w == 0:
        return np.zeros((size, size), dtype=np.uint8)
    
    scale = (size - 4) / max(h, w)  # small margin
    if scale <= 0:
        return np.zeros((size, size), dtype=np.uint8)
    
    new_w, new_h = int(w * scale), int(h * scale)
    
    # Ensure new dimensions are at least 1
    new_w = max(1, new_w)
    new_h = max(1, new_h)
    
    resized = cv2.resize(crop, (new_w, new_h), interpolation=cv2.INTER_AREA)

    canvas = np.zeros((size, size), dtype=np.uint8)
    y_off = (size - new_h) // 2
    x_off = (size - new_w) // 2
    canvas[y_off:y_off+new_h, x_off:x_off+new_w] = resized
    return canvas

def segment_characters(img: np.ndarray):
    """
    Returns list of (char_img_28x28, bbox) sorted left->right.
    bbox = (x, y, w, h) in original image coords.
    """
    gray = to_grayscale(img)
    th = binarize(gray)

    # remove small noise
    th = cv2.medianBlur(th, 3)

    contours, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes = []
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        # Increase minimum size threshold to avoid very small noise
        if w * h < 100 or w < 5 or h < 5:  # ignore tiny noise blobs
            continue
        boxes.append((x, y, w, h))

    if not boxes:
        return [], th
    
    # sort top-to-bottom, then left-to-right
    # Group by approximate row (allowing some vertical overlap)
    # Calculate average height for row grouping
    avg_height = np.mean([h for _, _, _, h in boxes])
    row_tolerance = avg_height * 0.6  # Characters in same row if y-difference < 60% of avg height
    
    # Sort by row (y-coordinate with tolerance), then by x-coordinate
    def sort_key(box):
        x, y, w, h = box
        # Group into rows based on y-coordinate
        row = int(y / row_tolerance)
        return (row, x)  # Sort by row first, then by x within row
    
    boxes.sort(key=sort_key)

    chars = []
    for (x, y, w, h) in boxes:
        try:
            char_crop = th[y:y+h, x:x+w]  # binary (inverted already)
            if char_crop.size == 0:
                continue
            char_28 = resize_and_center(char_crop, IMG_SIZE)
            chars.append((char_28, (x, y, w, h)))
        except Exception as e:
            # Skip problematic character segments
            continue

    return chars, th

def preprocess_image(img: np.ndarray) -> np.ndarray:
    """
    Preprocess a single image for model input.
    Converts to grayscale, binarizes, resizes/centers, and normalizes to [0, 1].
    Returns: (28, 28) float array in range [0, 1]
    """
    gray = to_grayscale(img)
    binary = binarize(gray)
    processed = resize_and_center(binary, IMG_SIZE)
    # Normalize to [0, 1]
    return processed.astype(np.float32) / 255.0
