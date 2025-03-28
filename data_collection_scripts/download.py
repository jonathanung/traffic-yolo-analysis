import kagglehub
import os
import shutil

"""
Downloads and copies the LISA Traffic Light Dataset to the data directory.

Returns the path where the dataset was copied.
"""
def download_and_copy_lisa() -> str:
    try:
        lisa_data_dir: str = "data/raw/lisa"
        os.makedirs(lisa_data_dir, exist_ok=True)
        
        print("Checking/Downloading LISA Traffic Light Dataset...")
        lisa_path: str = kagglehub.dataset_download("mbornoe/lisa-traffic-light-dataset")
        print(f"LISA dataset path: {lisa_path}")
        
        if not os.listdir(lisa_data_dir):
            print(f"Copying LISA dataset to {lisa_data_dir}...")
            for item in os.listdir(lisa_path):
                s: str = os.path.join(lisa_path, item)
                d: str = os.path.join(lisa_data_dir, item)
                if os.path.isdir(s):
                    shutil.copytree(s, d, dirs_exist_ok=True)
                else:
                    shutil.copy2(s, d)
            print("LISA dataset copied to data directory")
        else:
            print(f"LISA dataset already exists in {lisa_data_dir}")
            
        return lisa_data_dir
    except:
        raise Exception("Failed to download or copy LISA dataset. Check your Kaggle credentials.")

"""
Downloads and copies the Bosch Traffic Light Dataset to the data directory.

Returns the path where the dataset was copied.
"""
def download_and_copy_bosch() -> str:
    try:
        bosch_data_dir: str = "data/raw/bosch"
        os.makedirs(bosch_data_dir, exist_ok=True)
        
        print("Checking/Downloading Bosch Small Traffic Lights Dataset...")
        bosch_path: str = kagglehub.dataset_download("researcherno1/small-traffic-lights")
        print(f"Bosch dataset path: {bosch_path}")
        
        if not os.listdir(bosch_data_dir):
            print(f"Copying Bosch dataset to {bosch_data_dir}...")
            for item in os.listdir(bosch_path):
                s: str = os.path.join(bosch_path, item)
                d: str = os.path.join(bosch_data_dir, item)
                if os.path.isdir(s):
                    shutil.copytree(s, d, dirs_exist_ok=True)
                else:
                    shutil.copy2(s, d)
            print("Bosch dataset copied to data directory")
        else:
            print(f"Bosch dataset already exists in {bosch_data_dir}")
            
        return bosch_data_dir
    except:
        raise Exception("Failed to download or copy Bosch dataset. Check your Kaggle credentials.")

def main():
    try:
        os.makedirs("data/raw", exist_ok=True)
        
        lisa_dir: str = download_and_copy_lisa()
        bosch_dir: str = download_and_copy_bosch()
        
        print(f"Datasets downloaded and copied to:")
        print(f"- LISA: {lisa_dir}")
        print(f"- Bosch: {bosch_dir}")
    except Exception as e:
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    main()