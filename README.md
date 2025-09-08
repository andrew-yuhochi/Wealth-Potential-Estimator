# Wealth-Potential-Estimator


This is a application that estimates an individual's potential net worth by recognizing their face and comparing it to a dataset of well-known, high-net-worth individuals. The application leverages a pre-trained Vision Transformer model to extract a unique "face embedding" and then uses cosine similarity to find the most similar matches in a mock database. The final net worth is estimated based on the average net worth of the top matches.


## Architectural Decisions


This project was built with a focus on modularity, simplicity, and performance. Below are the key engineering decisions that shaped the application's architecture.


*1. Modular and Scalable Codebase*


The application logic is broken down into distinct, single-purpose Python files (```app.py```, ```config.py```, ```data_generation.py```, ```embedding_service.py```, ```similarity_search.py```). This design pattern ensures a clean, readable codebase where each component has a clear responsibility. This modularity makes the application easier to test, debug, and scale, as each component could be swapped out or updated independently in the future.


*2. Pre-trained Vision Transformer for Embeddings*


To handle the core task of image recognition, we selected a pre-trained Vision Transformer (ViT) model. Using a model from a well-known library like Hugging Face allows us to leverage state-of-the-art capabilities without the time and computational cost of training a model from scratch. The model extracts a high-dimensional feature vector, or "embedding," that accurately represents the unique features of a face.


*3. Vectorized Similarity Search*


For the core search functionality, we implemented cosine similarity from scratch using vectorized operations from the NumPy library. This approach was chosen to minimize dependencies and reduce the final container size. By performing calculations on entire arrays at once, this method is significantly more performant than iterating through each item, ensuring efficient search times even with larger datasets.


*4. Rapid Prototyping with Gradio*


The application's user interface is built with Gradio. This framework was selected for its simplicity and speed in building interactive web interfaces for machine learning models. Gradio automatically handles the complexities of a web server and provides a user-friendly public URL, making it an ideal choice for this project's API and demonstration purposes.


*5. Mock Dataset for Local Development*


A self-generated mock dataset was created for the purposes of development and testing. This approach allowed for rapid prototyping without the need to manage a real-world dataset. The mock data includes images, names, and a placeholder for net worth, providing a complete environment for end-to-end testing of the application's logic.


## How to Run the Solution


The easiest way to run this application is by using Docker. Docker provides a consistent, self-contained environment that includes all the necessary dependencies without requiring you to install them on your machine.


*Step 1: Build the Docker Image*

Make sure you have Docker installed and running on your system. Navigate to your project's root folder in the terminal and run the following command. This will build a Docker image named ```wealth-estimator:1.0``` based on the instructions in the ```Dockerfile```.

```docker build -t wealth-estimator:1.0```

*Step 2: Run the Docker Container*

Once the image is built, run the following command to start the application. The ```-p 7860:7860``` flag maps the container's internal port to your local machine, allowing you to access the Gradio interface.

```docker run -p 7860:7860 wealth-estimator:1.0```
