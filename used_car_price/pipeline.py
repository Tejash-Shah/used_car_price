import logging

from feature_engine.categorical_encoders import RareLabelCategoricalEncoder, CountFrequencyCategoricalEncoder
from feature_engine.missing_data_imputers import ArbitraryNumberImputer, CategoricalVariableImputer
from sklearn.pipeline import Pipeline
from xgboost import XGBRegressor

from used_car_price.processing import preprocessors

_logger = logging.getLogger(__name__)

used_car_price_pipeline = Pipeline(
    [
        (
            "preprocessing",
            preprocessors.Preprocessing()
        ),
        (
            "random_num_impute",
            ArbitraryNumberImputer(arbitrary_number=-9, variables=['year', 'odometer'])
        ),
        (
            "cylinder_impute",
            CategoricalVariableImputer(imputation_method='missing', fill_value='-1', variables=['cylinders'])
        ),
        (
            "categorical_impute",
            CategoricalVariableImputer(imputation_method='missing', fill_value='missing')
        ),
        (
            "rare_label_manufacturer",
            RareLabelCategoricalEncoder(tol=0.01, variables='manufacturer', n_categories=5, replace_with='rare')
        ),
        (
            "rare_label_cylinder",
            RareLabelCategoricalEncoder(tol=0.01, variables='cylinders', n_categories=5, replace_with='-1')
        ),
        (
            "rare_label_condition",
            RareLabelCategoricalEncoder(tol=0.07, variables=['condition'], n_categories=3, replace_with='rare')
        ),
        (
            'rare_label_type',
            RareLabelCategoricalEncoder(tol=0.04, variables=['type'], n_categories=3, replace_with='rare')
        ),
        (
            'rare_label_paint_color',
            RareLabelCategoricalEncoder(tol=0.05, variables=['paint_color'], n_categories=3, replace_with='rare')
        ),
        (
            'rare_label_transmission',
            RareLabelCategoricalEncoder(tol=0.05, variables=['transmission'], n_categories=3, replace_with='rare')
        ),
        (
            'frequency_encode',
            CountFrequencyCategoricalEncoder(encoding_method='frequency')
        ),
        (
            'xgboost_model',
            XGBRegressor()
        )

    ]
)
