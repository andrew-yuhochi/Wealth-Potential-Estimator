import gradio as gr
import numpy as np
import os
import threading
import time
import tempfile
from transformers import AutoConfig
from PIL import Image

from embedding_service import get_image_embedding
from similarity_search import find_most_similar_celebrities, load_dataset
from data_generation import create_mock_dataset
from config import MOCK_DATASET_PATH, MODEL_ID, SIMILARITY_THRESHOLD

def recognize_and_estimate(image):
    """
    Accepts an image, gets its embedding, finds similar celebrities,
    and returns an estimated net worth and a gallery of matches.
    """
    if image is None:
        return "No image provided.", None

    try:
        # Get the embedding for the uploaded image
        query_embedding = get_image_embedding(image)

        # Use the embedding to find the most similar celebrities
        matches = find_most_similar_celebrities(query_embedding)

        # Calculate the estimated potential net worth
        if not matches:
            estimated_net_worth = "No Matches Found"
            return estimated_net_worth, None
        else:
            # Take the average net worth of the top matches
            average_net_worth = np.mean([match['profile']['net_worth_USD'] for match in matches])
            estimated_net_worth = f"${average_net_worth:,.2f}"

        # Create a list of tuples for the Gradio Gallery component
        # Each tuple is (image_path, caption)
        gallery_matches = []
        for match in matches:
            profile = match['profile']
            score = match['score']
            caption = (
                f"{profile['name']}\n"
                f"Similarity: {score:.2%}"
            )
            gallery_matches.append((profile['image_path'], caption))
        
        return estimated_net_worth, gallery_matches

    except Exception as e:
        print(f"An error occurred: {e}")
        return f"An error occurred: {e}", None


if __name__ == "__main__":
    # Ensure the model path is valid
    try:
        AutoConfig.from_pretrained(MODEL_ID)
        print(f"Model ID '{MODEL_ID}' validated successfully.")
    except Exception as e:
        raise ValueError(f"Invalid model ID configured: '{MODEL_ID}'. Error: {e}")

    # Ensure the mock dataset exists before we start the app
    if not os.path.exists(MOCK_DATASET_PATH):
        print("Mock dataset not found. Generating it now...")
        create_mock_dataset()
        print("Dataset generation complete.")
    
    # Ensure the app is started as expected
    print("\nRunning a quick end-to-end test with a sample image...")
    # Use the first image from our mock dataset to guarantee a valid test
    MOCK_DATASET = load_dataset(MOCK_DATASET_PATH)
    sample_image_path = MOCK_DATASET[0]['image_path']
    
    try:
        test_result = recognize_and_estimate(sample_image_path)
        print("Test Complete. Sample Output:")
        print(f"Estimated Net Worth: {test_result[0]}")
        if test_result[1]:
            print(f"Found {len(test_result[1])} matches.")
            print(f"Top Match: {test_result[1][0][1]}")
        else:
            print("No matches found in test.")
    except Exception as e:
        print(f"Test Failed: An error occurred during the end-to-end test: {e}")

    print("--- Validation Complete ---")

    # Create the Gradio Interface
    iface = gr.Interface(
        fn=recognize_and_estimate,
        inputs=gr.Image(type="filepath", label="Upload an image"),
        outputs=[
            gr.Textbox(label="Estimated Net Worth"),
            gr.Gallery(
                label="Top Celebrity Matches",
                columns=[3],
                rows=[1],
                height="auto",
                object_fit="contain",
            )
        ],
        title="Wealth Potential Estimator",
        description=(
            "Upload your selfie to get an estimated potential net worth."
        )
    )

    # Launch the Gradio app with `share=True` to create a public URL
    iface.launch(share=True)