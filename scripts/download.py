import kagglehub
import os

# Create data directory if it doesn't exist
os.makedirs("data/raw", exist_ok=True)

# Download LISA Traffic Light Dataset
print("Downloading LISA Traffic Light Dataset...")
lisa_path = kagglehub.dataset_download("mbornoe/lisa-traffic-light-dataset")
print(f"LISA dataset downloaded to: {lisa_path}")

# Download Bosch Traffic Light Dataset
print("Downloading Bosch Small Traffic Lights Dataset...")
bosch_path = kagglehub.dataset_download("researcherno1/small-traffic-lights")
print(f"Bosch dataset downloaded to: {bosch_path}")