import tensorflow as tf
from transformers import AutoFeatureExtractor, TFViTModel
from PIL import Image
import numpy as np
import os
from config import MODEL_ID, TEST_IMAGE_PATH

# 0. Load the model and feature extractor
feature_extractor = AutoFeatureExtractor.from_pretrained(MODEL_ID)
model = TFViTModel.from_pretrained(MODEL_ID, from_pt=True)

def get_image_embedding(image_file_object):
    """
    Extracts a feature embedding from an image using a pre-trained Vision Transformer.

    This function is designed to take a file-like object directly, as received from
    a REST API upload.
    It will then preprocesses the input image, and then passes it through the model 
    to extract a numerical feature vector.

    Args:
        image_file_object: A file-like object (e.g., from FastAPI's UploadFile).

    Returns:
        np.ndarray: A Numpy array representing the image's feature embedding. 
                    The shape of the array will be (1, 768).
    """
    try:
        # 1. Load and process the image
        image = Image.open(image_file_object).convert("RGB")
    
        # Apply preprocessing using the feature extractor
        inputs = feature_extractor(images=image, return_tensors="tf")
    
        # 2. Get the embedding vector
        # We need to access the hidden states as embedding instead of the classification level
        outputs = model(**inputs, output_hidden_states=True)
    
        # Extract the embedding from the last hidden state of the [CLS] token
        embedding = outputs.last_hidden_state[:, 0, :].numpy()
    
        # 3. Return the embedding
        return embedding

    except Exception as e:
        # Catch any errors that might occur during image processing
        raise ValueError(f"An error occurred while processing the uploaded file: {e}")

if __name__ == "__main__":
    # Check if the test image exists
    if not os.path.exists(TEST_IMAGE_PATH):
        print(f"Test image not found at '{TEST_IMAGE_PATH}'. Skipping functional test.")
    else:
        # Test the function with a sample image
        try: 
            with open(TEST_IMAGE_PATH, 'rb') as image_file_object:
                embedding = get_image_embedding(image_file_object)
    
            # Expected embedding should be (1, 768) dimension 
            assert embedding.shape == (1, 768), f"Expected shape (1, 768) but got {embedding.shape}"
        
            print("\nEmbedding shape:", embedding.shape)
            print("Embedding type:", type(embedding))
            print("Function executed successfully and output shape is correct!")
    
        except Exception as e:
            print(f"\nAn error occurred during functional testing: {e}")
            raise