import pandas as pd
import joblib
from used_car_price.config import config
from sklearn.pipeline import Pipeline


def load_dataset(file_name: str, **kwargs) -> pd.DataFrame:
    _data = pd.read_csv(f"{config.RAW_DATASET_DIR}/{file_name}", **kwargs)
    return _data


def save_dataset(df: pd.DataFrame, file_name: str, **kwargs) -> None:
    df.to_csv(f"{config.RAW_DATASET_DIR}/{file_name}", **kwargs)


# def save_model(model_name: str, model) -> None:
#     joblib.dump(model, f"{config.TRAINED_MODELS_DIR}\\{model_name}.joblib")


def save_pipeline(pipeline: Pipeline, pipeline_name: str) -> None:
    joblib.dump(pipeline,  f"{config.TRAINED_MODELS_DIR}\\{pipeline_name}.pkl")


def load_pipeline(file_name: str) -> Pipeline:
    model = f"{config.TRAINED_MODELS_DIR}/{file_name}.pkl"
    return joblib.load(filename=model)


def drop_duplicates_in_df(df: pd.DataFrame, **kwargs):
    return df.drop_duplicates(**kwargs).reset_index(drop=True)


def remove_dirty_target(df: pd.DataFrame):
    return df[(df[config.TARGET[0]] >= config.TARGET_MIN) & (df[config.TARGET[0]] <= config.TARGET_MAX)]
