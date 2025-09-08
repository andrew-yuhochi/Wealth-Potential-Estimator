# Use an official Python runtime as a parent image
FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file into the container at /app
# We do this first to leverage Docker's caching
COPY requirements.txt .

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code into the container
COPY . .

# Expose the port that Gradio runs on
EXPOSE 7860

# Command to run the application
# We use `python -u` for unbuffered output, which is helpful for logs
CMD ["python", "-u", "app.py"]