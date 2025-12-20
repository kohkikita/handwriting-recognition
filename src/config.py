
IMG_SIZE = 28
EMNIST_SPLIT = "balanced"
NUM_CLASSES = 47  # EMNIST balanced has 47 classes

# NOTE: EMNIST Balanced includes digits + uppercase letters in a specific order.
# You should verify label->char order (quick check in code after loading dataset).
LABELS_FALLBACK = [str(i) for i in range(10)] + [chr(ord('A') + i) for i in range(26)]
