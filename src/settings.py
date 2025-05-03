"""Persistent settings for the project"""

import os

CONFIG_PATH = "configs/config.yaml"

TRAIN_LOG_DIR = "runs/train"
INFER_LOG_DIR = "runs/infer"

CLASS_ENCODING = {
    "background": [0, 0, 0],
    "building": [0, 0, 128],
    "human": [0, 64, 64],
    "tree": [0, 128, 0],
    "low_vegetation": [0, 128, 128],
    "moving_vehicle": [128, 0, 64],
    "road": [128, 64, 128],
    "static_vehicle": [192, 0, 192],
}

COMET_API_KEY = os.environ.get("COMET_API_KEY", None)
