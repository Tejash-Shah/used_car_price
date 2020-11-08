import numpy as np
import pandas as pd
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.pipeline import Pipeline

from used_car_price.config import config
from used_car_price.processing import load_pipeline


def _eval_metrics(actual, pred):
    r2 = np.round(r2_score(y_true=actual, y_pred=pred) * 100, 2)
    rmse = int(np.sqrt(mean_squared_error(y_true=actual, y_pred=pred)))
    mae = int(mean_absolute_error(y_true=actual, y_pred=pred))
    return r2, rmse, mae


def _predict_and_score(trained_pipeline: Pipeline, df: pd.DataFrame) -> None:
    df['predictions'] = trained_pipeline.predict(df[config.FEATURES])
    print(_eval_metrics(actual=df[config.TARGET], pred=df['predictions']))
    # mlflow.log_metric("test r2",r2)


def main() -> None:
    # test = load_dataset(file_name=config.TEST_DATA_FILE)
    # test = drop_duplicates_in_df(df=test, subset=None)
    # test = remove_dirty_target(df=test)
    trained_pipeline = load_pipeline(file_name=config.PIPELINE_NAME)
    print(trained_pipeline.named_steps['xgboost_model'])
    # _predict_and_score(trained_pipeline=trained_pipeline, df=test)


if __name__ == '__main__':
    main()
