#!/bin/bash

# Test all models
# python3 ./test_models.py --model_version v8 --output_dir results --save_txt --save_imgs
# python3 ./test_models.py --model_version v5 --output_dir results --save_txt --save_imgs
# python3 ./test_models.py --model_version v3 --output_dir results --save_txt --save_imgs      

python3 ./data_collection_scripts/yolo8_test.py
python3 ./data_collection_scripts/yolo5_test.py
python3 ./data_collection_scripts/yolo3_test.py