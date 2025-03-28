#! /bin/bash

# This script downloads the data, and does the entire split, including data processing, training, and testing
# Evaluation scripts will be run separately

# Download the data
python ./data_collection_scripts/download_data.py

# Process the data
python ./data_collection_scripts/process_data.py

# Data is already split by default, as given by authors of LISA

# Train the models
sh ./train_all_models.sh

# Test the models
sh ./test_all_models.sh

