#!/bin/bash

# Test all models
python3 ./test_models.py --model_version v8 --output_dir results      
python3 ./test_models.py --model_version v5 --output_dir results      
python3 ./test_models.py --model_version v3 --output_dir results      