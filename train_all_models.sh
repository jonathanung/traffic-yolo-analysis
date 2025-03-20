#! /bin/bash

# Train all models - note this is for CUDA based systems
python ./scripts/train_lisa.py --model_version=v3 --device=0
python ./scripts/train_lisa.py --model_version=v5 --device=0
python ./scripts/train_lisa.py --model_version=v8 --device=0
