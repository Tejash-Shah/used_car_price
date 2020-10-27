import math
import shap
import pandas as pd
from sklearn.model_selection import cross_validate
from feature_engine.outlier_removers import OutlierTrimmer
from used_car_price import pipeline
from used_car_price.config import config
from used_car_price.processing import load_dataset, save_pipeline, remove_dirty_target, drop_duplicates_in_df
from sklearn.pipeline import Pipeline
from xgboost import plot_importance
from matplotlib import pyplot as plt
# import mlflow
import numpy as np
from mlflow import log_metric, log_param, log_artifacts
from pprint import pprint
import logging


def _cross_validate_pipeline(df: pd.DataFrame) -> None:

    scores = cross_validate(estimator=pipeline.used_car_price_pipeline,
                            X=df[config.FEATURES],
                            y=df[config.TARGET].values.ravel(),
                            cv=5,
                            scoring=('r2', 'neg_mean_squared_error', 'neg_mean_absolute_error'),
                            return_train_score=True,
                            return_estimator=False)

    print(f"Train r2: {round(scores['train_r2'].mean() * 100, 1)}")
    print(f"Valid r2: {round(scores['test_r2'].mean() * 100, 1)}")
    print(f"Train MSE: {int(math.sqrt(-scores['train_neg_mean_squared_error'].mean() * 100))}")
    print(f"Valid MSE: {int(math.sqrt(-scores['test_neg_mean_squared_error'].mean() * 100))}")


def feature_importance_xgboost(trained_pipeline: Pipeline):
    # feat_importance = trained_pipeline[config.ESTIMATOR_NAME].feature_importances_
    plot_importance(trained_pipeline[config.ESTIMATOR_NAME])
    plt.show()


def shap_plot(trained_pipeline: Pipeline, df: pd.DataFrame):
    explainer = shap.TreeExplainer(trained_pipeline[config.ESTIMATOR_NAME])
    transformed_data = trained_pipeline.fit_transform(df[config.FEATURES])
    print(transformed_data.shape)
    shap_values = explainer.shap_values(transformed_data)
    # summarize the effects of all the features
    shap.summary_plot(shap_values, transformed_data)


def _train_model(df: pd.DataFrame) -> Pipeline:
    return pipeline.used_car_price_pipeline.fit(X=df[config.FEATURES], y=df[config.TARGET])


def main() -> None:
    train = load_dataset(file_name=config.TRAIN_DATA_FILE, usecols=config.FEATURES+config.TARGET)
    train = drop_duplicates_in_df(df=train, subset=None)
    train = remove_dirty_target(df=train)
    train = drop_duplicates_in_df(df=train, subset=['vin'], keep='first')
    _cross_validate_pipeline(df=train)
    trained_model = _train_model(df=train)
    feature_importance_xgboost(trained_pipeline=trained_model)
    # shap_plot(trained_pipeline=trained_model, df=train)
    # save_pipeline(pipeline=trained_model, pipeline_name=config.PIPELINE_NAME)


if __name__ == '__main__':
    main()
