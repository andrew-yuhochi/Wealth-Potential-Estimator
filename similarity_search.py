import json
import numpy as np
from numpy.linalg import norm
from config import MOCK_DATASET_PATH, SIMILARITY_THRESHOLD, TopK

def load_dataset(dataset_path):
    """
    Loads the celebrity profiles dataset from a JSON file.

    Args:
        dataset_path (str): The file path to the JSON dataset.

    Returns:
        list: A list of dictionaries, where each dictionary represents a celebrity profile
              and includes the name, net worth, image path, and embedding.
    """
    try:
        with open(dataset_path, "r") as f:
            data = json.load(f)
            return data
    except FileNotFoundError:
        raise FileNotFoundError(f"Dataset not found at '{dataset_path}'. Please run the data generation script first.")
    except json.JSONDecodeError:
        raise ValueError(f"Error decoding JSON from '{dataset_path}'. The file may be corrupted.")

def find_most_similar_celebrities(query_embedding):
    """
    Finds and returns the most similar celebrities from the mock dataset.

    This function uses a vectorized approach for highly efficient similarity
    calculations. It calculates the cosine similarity between the query embedding
    and all embeddings in the dataset at once.

    Args:
        query_embedding (np.ndarray): The embedding of the unknown person to search for.

    Returns:
        list: A list of dictionaries, sorted by similarity score in descending order.
              Each dictionary contains the celebrity's profile and their similarity score.
              Returns an empty list if no matches are found above the threshold.
    """
    # Load the mock dataset and extract embeddings
    celebrity_data = load_dataset(MOCK_DATASET_PATH)
    db_embeddings = np.array([profile["embedding"] for profile in celebrity_data])
    
    # Ensure query embedding and db embeddings have the same dimension
    if query_embedding.shape[1] != db_embeddings.shape[1]:
        raise ValueError("Query embedding dimension does not match dataset embedding dimension.")

    # 1. Vectorized Dot Product
    dot_products = np.dot(query_embedding, db_embeddings.T).flatten()

    # 2. Vectorized Norm Calculation
    db_norms = norm(db_embeddings, axis=1)

    # 3. Vectorized Cosine Similarity
    query_norm = norm(query_embedding)
    
    # Avoid division by zero
    similarity_scores = np.zeros_like(dot_products)
    non_zero_norms = (query_norm * db_norms) != 0
    similarity_scores[non_zero_norms] = dot_products[non_zero_norms] / (query_norm * db_norms[non_zero_norms])

    # Combine profiles with their scores
    matches = []
    for i, score in enumerate(similarity_scores):
        if score >= SIMILARITY_THRESHOLD: 
            matches.append({
                "profile": celebrity_data[i],
                "score": score
            })

    # Sort the matches by score in descending order
    matches.sort(key=lambda x: x["score"], reverse=True)
    
    # Return only the top K matches, as defined in our config
    return matches[:TopK]

# This is a sample usage to test the function's logic
if __name__ == "__main__":
    from embedding_service import get_image_embedding
    from config import TEST_IMAGE_PATH

    print("Running a test search...")
    try:
        # Get the embedding for our test image
        with open(TEST_IMAGE_PATH, 'rb') as f:
            test_embedding = get_image_embedding(f)
        
        # Find the most similar celebrities
        similar_celebrities = find_most_similar_celebrities(test_embedding)
        
        print("\n--- Search Results ---")
        if similar_celebrities:
            for match in similar_celebrities:
                profile = match["profile"]
                score = match["score"]
                print(f"Match: {profile['name']}, Similarity Score: {score:.4f}")
        else:
            print("No close matches found. The test image may not match any celebrity in the dataset.")
            
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please ensure you have run the data generation script and have a test image in place.")
    except Exception as e:
        print(f"An unexpected error occurred during the test: {e}")