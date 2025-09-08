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

## How to Use the Application

Using the Wealth Potential Estimator is simple and straightforward.

Upload Your Image: In the "Upload an image" box on the left, you can either click the upload icon to select an image from your computer or drag and drop an image file directly into the box. You aslo also take and upload a new picture with your camera. 

Submit Your Request: Once your image appears in the preview pane, click the orange "Submit" button. The application will process your image and run the estimation.

View Your Results: On the right-hand side, two key pieces of information will be displayed:

Estimated Net Worth: A dollar value representing your estimated net worth based on your top celebrity matches.

Top Celebrity Matches: A gallery of the three celebrities from our mock dataset who have the most similar facial features to your own. You can further click into eacah of them to understand more about the celebrity's name and a similarity score.

## File Contents

```Dockerfile```

This file contains the instructions for building the Docker image.

```requirements.txt```

This file lists the Python packages and their versions required by the application.

```app.py```

This is the main script that defines the Gradio interface and orchestrates the application logic.

```config.py```

This file holds all the application's configurable settings, such as model paths and data file locations.

```data_generation.py```

This script is responsible for creating the mock dataset, including generating and storing the embeddings.

```embedding_service.py```

This file contains the core logic for loading the Vision Transformer model and generating a numerical embedding from an image.

```similarity_search.py```

This script contains the logic for loading the dataset and performing a vectorized cosine similarity search.
