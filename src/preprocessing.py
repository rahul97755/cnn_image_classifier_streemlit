import numpy as np
from tensorflow.keras.preprocessing import image

def preprocess_image(img):
    """Resize, normalize and expand dimensions for CNN input"""
    img = img.resize((32, 32))
    img_array = image.img_to_array(img)
    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array
