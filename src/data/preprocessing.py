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
    scale = (size - 4) / max(h, w)  # small margin
    new_w, new_h = int(w * scale), int(h * scale)
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
        if w * h < 50:  # ignore tiny noise blobs
            continue
        boxes.append((x, y, w, h))

    # sort left-to-right
    boxes.sort(key=lambda b: b[0])

    chars = []
    for (x, y, w, h) in boxes:
        char_crop = th[y:y+h, x:x+w]  # binary (inverted already)
        char_28 = resize_and_center(char_crop, IMG_SIZE)
        chars.append((char_28, (x, y, w, h)))

    return chars, th
