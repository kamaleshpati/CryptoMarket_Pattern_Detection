import json
import os

CONFIG_PATH = os.path.join("data", "configuration", "config.json")

def load_config():
    with open(CONFIG_PATH, "r") as f:
        return json.load(f)

_config = load_config()

# Assign constants for easy access
MODEL_PATH = _config["MODEL_PATH"]
FEATURE_PATH = _config["FEATURE_PATH"]
RAW_DATA_PATH = _config["RAW_DATA_PATH"]
CONFIDENCE_THRESHOLD = _config["CONFIDENCE_THRESHOLD"]
MIN_VALID_PATTERNS = _config["MIN_VALID_PATTERNS"]
RULE_REPORT_PATH = _config["RULE_REPORT_PATH"]
ML_REPORT_PATH = _config["ML_REPORT_PATH"]
DATA_PATH = _config["DATA_PATH"]
OUTPUT_DIR = _config["OUTPUT_DIR"]
