# Model Path
MODEL_ID = "google/vit-base-patch16-224"

# Similarity and Search Configuration
TopK = 3 # Number of result to be returned
SIMILARITY_THRESHOLD = 0 # Minimum threshold to consider a match

# Dataset Configuration
MOCK_DATASET_PATH = "./Celebrity_Image/mock_profiles.json" 
MOCK_PROFILES = [
  {"name": "Taylor Swift", "net_worth_USD": 1300000000, "image_path": "./Celebrity_Image/TaylorSwift.jpg"},
  {"name": "Jensen Huang", "net_worth_USD": 91000000000, "image_path": "./Celebrity_Image/JensenHuang.jpg"},
  {"name": "Elon Musk", "net_worth_USD": 200000000000, "image_path": "./Celebrity_Image/ElonMusk.jpg"},
  {"name": "Andrew Ng", "net_worth_USD": 80000000, "image_path": "./Celebrity_Image/AndrewNg.jpg"},
  {"name": "Oprah Winfrey", "net_worth_USD": 2800000000, "image_path": "./Celebrity_Image/OprahWinfrey.jpg"}
  ]

# Test Configuration
# This is for internal testing purposes only and will not be used in the production API
TEST_IMAGE_PATH = "./Celebrity_Image/AndrewNg.jpg"