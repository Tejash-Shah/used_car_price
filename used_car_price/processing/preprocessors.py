from sklearn.base import BaseEstimator, TransformerMixin
from feature_engine.outlier_removers import OutlierTrimmer

class Preprocessing(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        X = X.copy()
        X.loc[X['vin'].isna(), 'vin']=0
        X.loc[~X['vin'].isna(), 'vin']=1
#         X['cylinders'] = X['cylinders'].str.replace('cylinders','')
#         X.loc[X['cylinders']=='other', 'cylinders'] = '99'
        X['vin'] = X['vin'].astype('int')
        return X


class ChangeColType(BaseEstimator, TransformerMixin):
    def __init__(self, col, col_type):
        self.col = col
        self.col_type = col_type

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        X[self.col] = X[self.col].astype(self.col_type)
        return X

# class TrimTarget(BaseEstimator, TransformerMixin):
#
#     def __init__(self):
#         pass
#
#     def fit(self, X, y):



