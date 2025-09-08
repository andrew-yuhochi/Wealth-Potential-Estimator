import json
import os
import numpy as np
from embedding_service import get_image_embedding 
from config import MOCK_DATASET_PATH, MOCK_PROFILES

def create_mock_dataset():
    """
    Generates a mock dataset of celebrity profiles with image embeddings.
    This is a one-off execution. 

    This function reads a list of celebrity image files, processes them
    to generate a numerical embedding, and saves the data to a JSON file.

    Args:
      None. All information are extracted from configuration.py directly.

    Returns:
        None. This function does not return any value. It saves the
        generated data to a file on disk.

    Note:
        The output file will be a JSON array of objects, where each object has
        the following format:
        {
          "name": str,
          "net_worth_USD": float,
          "image_path": str,
          "embedding": list of floats
        }
    """

    print("Starting the data generation process...")
    
    mock_data = []
    success_count = 0
    error_count = 0

    # Use a try-finally block to ensure a summary is always reported
    try:
        for profile in MOCK_PROFILES:
            try:
                print(f"Processing image for {profile['name']}...")
                
                # Open the image file as a file-like object
                with open(profile["image_path"], 'rb') as f:
                    # Use the imported function to get the embedding
                    embedding = get_image_embedding(f)
                    
                    # Convert the numpy array to a list for JSON serialization
                    profile["embedding"] = embedding.tolist()[0] 
                    mock_data.append(profile)
                    success_count += 1
                    
                print(f"Successfully created embedding for {profile['name']}.")
                
            except FileNotFoundError:
                print(f"Error: The image file for {profile['name']} was not found at '{profile['image_path']}'.")
                error_count += 1
            except Exception as e:
                print(f"An unexpected error occurred while processing {profile['name']}. Error: {e}")
                error_count += 1
    finally:
        # Report summary statistics
        print("\n--- Data Generation Summary ---")
        print(f"Total profiles to process: {len(MOCK_PROFILES)}")
        print(f"Successfully processed: {success_count}")
        print(f"Failed to process: {error_count}")

    # Check if the data directory exists, if not, create it
    os.makedirs(os.path.dirname(MOCK_DATASET_PATH), exist_ok=True)

    # Save the complete mock dataset to a JSON file
    with open(MOCK_DATASET_PATH, "w") as f:
        json.dump(mock_data, f, indent=4)
        
    print(f"\nData generation completed successfully!")
    print(f"Mock dataset saved to '{MOCK_DATASET_PATH}'.")

if __name__ == "__main__":
    create_mock_dataset()