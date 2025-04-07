# Setting up for Kaggle

### Register for Kaggle Account
https://www.kaggle.com/

### Install kagglehub
```bash
pip install kagglehub
```

### Set up Kaggle API credentials
1. Create a Kaggle account if you don't already have one at kaggle.com
2. Generate an API token:
    - Go to your Kaggle account settings: https://www.kaggle.com/account
    - Scroll down to the "API" section
    - Click "Create New API Token"
    - This will download a kaggle.json file containing your credentials
3. Place your credentials file:
    - Create a .kaggle directory in your home folder:
    ```bash
    mkdir ~/.kaggle
    ```
    - Move the downloaded kaggle.json file to this directory:
    ```bash
    mv kaggle.json ~/.kaggle/
    ```
4. Set proper permissions (important for security):
    ```bash
    chmod 600 ~/.kaggle/kaggle.json
    ```

## LISA Traffic Light Dataset

Get this dataset from kaggle

[Link](https://www.kaggle.com/datasets/mbornoe/lisa-traffic-light-dataset)

```python
import kagglehub

# Download latest version
path = kagglehub.dataset_download("mbornoe/lisa-traffic-light-dataset")

print("Path to dataset files:", path)
```

## Bosch Traffic Light Dataset

* Note: we ended up not using Bosch due to time limitations

This dataset is also from kaggle

[Link](https://www.kaggle.com/datasets/researcherno1/small-traffic-lights)

```python
import kagglehub

# Download latest version
path = kagglehub.dataset_download("researcherno1/small-traffic-lights")

print("Path to dataset files:", path)
```