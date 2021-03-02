import pathlib
import used_car_price

# data
RAW_DATA_FILE = "data.csv.zip"
TRAIN_DATA_FILE = "train.csv.gz"
TEST_DATA_FILE = "test.csv.gz"

# Location
PACKAGE_ROOT = pathlib.Path(used_car_price.__file__).resolve().parent
TRAINED_MODELS_DIR = f"{PACKAGE_ROOT}\\trained_models"
RAW_DATASET_DIR = f"{PACKAGE_ROOT}\\datasets\\01_raw"
INTERMEDIATE_DATASET_DIR = f"{PACKAGE_ROOT}\\datasets\\02_intermediate"
PROCESSED_DATASET_DIR = f"{PACKAGE_ROOT}\\datasets\\03_processed"

# variables
TARGET = ["price"]

FEATURES = ['year',
            'manufacturer',
            'condition',
            'cylinders',
            'fuel',
            'odometer',
            'title_status',
            'transmission',
            'vin',
            'drive',
            'size',
            'type',
            'paint_color',
            'state'
            ]

RANDOM_STATE = 42

PIPELINE_NAME = 'pipeline'

TARGET_MIN = 1_000
TARGET_MAX = 57_260

ESTIMATOR_NAME = 'xgboost_model'
