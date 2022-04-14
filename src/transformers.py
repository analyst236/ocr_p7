import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler, LabelEncoder, PolynomialFeatures, OneHotEncoder, StandardScaler, FunctionTransformer, RobustScaler, OrdinalEncoder
from sklearn.neighbors import NearestNeighbors, KNeighborsClassifier
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.base import BaseEstimator, TransformerMixin


class FormatTransformer(BaseEstimator, TransformerMixin):
    """

    """
    num_attrs = []
    cat_attrs = []

    cols = []
    encoders = {}

    encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=np.nan)

    def __init__(self, verbose=0):
        self.verbose = verbose

    def log(self, message):
        if self.verbose !=0:
            print(message)

    def fit(self, X, y=None):
        self.log("fit formatter")
        self.cols= X.select_dtypes(include=['category', 'object']).columns
        self.encoder.fit(X[self.cols])

        return self

    def transform(self, X):
        self.log("transform formatter")
        X_ = X.copy()
        # VALEURS ABERRANTES
        X_['DAYS_EMPLOYED_ANOM'] = X_["DAYS_EMPLOYED"] == 365243
        X_["DAYS_EMPLOYED_ANOM"] = X_["DAYS_EMPLOYED_ANOM"].astype("int")

        # Replace the anomalous values with nan
        X_['DAYS_EMPLOYED'].replace({365243: np.nan}, inplace = True)
        # Traitement des valeurs négatives
        X_['DAYS_BIRTH'] = abs(X_['DAYS_BIRTH'])

        X_[self.cols] = self.encoder.transform(X_[self.cols])

        for col in [c for c in self.cols if c not in X_.columns]:
            X_[col] = 0

        self.log(f"{X_.shape}")
        return X_


class ImputerTransformer(BaseEstimator, TransformerMixin):
    imputer = SimpleImputer(strategy='median')

    def __init__(self, verbose=0):
        self.verbose = verbose

    def log(self, message):
        if self.verbose != 0:
            print(message)

    def fit(self, X, y=None):
        self.log("fit imputer")
        self.imputer.fit(X)
        return self

    def transform(self, X):
        self.log("transform imputer")
        X_ = X.copy()
        X_ = pd.DataFrame(self.imputer.transform(X_), index=X_.index, columns=X_.columns)
        self.log(X_.shape)
        return X_


class FeatureTransformer(BaseEstimator, TransformerMixin):

    def __init__(self, verbose=0):
        self.verbose = verbose

    def log(self, message):
        if self.verbose != 0:
            print(message)

    def fit(self, X, y=None):
        self.log("fit feature")
        return self

    def transform(self, X):
        self.log("transform feature")
        X_ = X.copy()
        # CREATION DE VARIABLE
        # CREDIT_INCOME_PERCENT: the percentage of the credit amount relative to a client's income
        # ANNUITY_INCOME_PERCENT: the percentage of the loan annuity relative to a client's income
        # CREDIT_TERM: the length of the payment in months (since the annuity is the monthly amount due
        # DAYS_EMPLOYED_PERCENT: the percentage of the days employed relative to the client's age
        # Again, thanks to Aguiar and his great script for exploring these features.
        X_['CREDIT_INCOME_PERCENT'] = X_['AMT_CREDIT'] / X_['AMT_INCOME_TOTAL']
        X_['ANNUITY_INCOME_PERCENT'] = X_['AMT_ANNUITY'] / X_['AMT_INCOME_TOTAL']
        X_['CREDIT_TERM'] = X_['AMT_ANNUITY'] / X_['AMT_CREDIT']
        X_['DAYS_EMPLOYED_PERCENT'] = X_['DAYS_EMPLOYED'] / X_['DAYS_BIRTH']
        self.log(X_.shape)
        return X_


class ScallerTransformer(BaseEstimator, TransformerMixin):
    # scaller = MinMaxScaler(feature_range = (0, 1))
    scaller = MinMaxScaler()

    def __init__(self, verbose=0):
        self.verbose = verbose

    def log(self, message):
        if self.verbose != 0:
            print(message)

    def fit(self, X, y=None):
        self.log("fit scaller")
        self.scaller.fit(X)
        return self

    def transform(self, X):
        self.log("transform scaller")
        X_ = X.copy()

        X_ = pd.DataFrame(self.scaller.transform(X_), index=X_.index, columns=X_.columns)
        self.log(X_.shape)
        return X_


class KNeighborsTransformer(BaseEstimator, TransformerMixin):
    # scaller = MinMaxScaler(feature_range = (0, 1))
    knn = KNeighborsClassifier(300)
    cols = []
    y_ = []

    def __init__(self,
                 apply_on=['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3', 'AMT_ANNUITY', 'AMT_CREDIT'],
                 verbose=0):
        self.verbose = verbose
        self.cols = apply_on

    def log(self, message):
        if self.verbose != 0:
            print(message)

    def fit(self, X, y):
        self.log("fit KNN")
        self.y_ = y
        self.knn.fit(X[self.cols], y)
        return self

    def transform(self, X):
        self.log("transform KNN")
        X_ = X.copy()

        X_['KNN_300'] = [self.y_.iloc[ele].mean() for ele in self.knn.kneighbors(X_[self.cols])[1]]

        self.log(X_.shape)
        return X_


class MetaTransformer(BaseEstimator, TransformerMixin):

    def __init__(self, verbose=0):
        self.verbose = verbose
        self.preprocessor_pipeline = Pipeline(steps=[
            ('format', FormatTransformer(verbose=self.verbose)),
            ('impute', ImputerTransformer(verbose=self.verbose)),
            ('feature', FeatureTransformer(verbose=self.verbose)),
            ('scale', ScallerTransformer(verbose=self.verbose)),
            ('knn', KNeighborsTransformer(verbose=self.verbose)),
        ])

    def log(self, message):
        if self.verbose != 0:
            print(message)

    def fit(self, X, y):
        self.log("meta _ fit transformers")
        self.preprocessor_pipeline.fit(X, y)
        return self

    def transform(self, X):
        self.log("meta _ transform X")
        self.log(X.shape)
        X_ = X.copy()
        result = self.preprocessor_pipeline.transform(X_)
        self.log(result.shape)
        return result

    def inverse_transform(self, Xt):
        # print("inverse transform ...")
        return self.preprocessor_pipeline.inverse_transform(Xt)


