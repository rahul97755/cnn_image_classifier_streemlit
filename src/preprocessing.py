import numpy as np
from PIL import Image

def preprocess_image(image: Image.Image, target_size=(32, 32)):
    """Resize, normalize and expand image dimensions."""
    img = image.resize(target_size)
    img_array = np.array(img) / 255.0
    return np.expand_dims(img_array, axis=0)
