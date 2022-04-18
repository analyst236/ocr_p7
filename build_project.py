import pandas as pd
import settings as conf
import logging
import logging.config
import os
import glob
from sklearn import metrics
from sklearn.model_selection import train_test_split, GridSearchCV
from src.transformers import MetaTransformer
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import SMOTE
from scipy.stats import randint, uniform
import lightgbm as lgb
import pickle


logging.config.fileConfig('logging.conf')


def load_raw_data(raw_data_path, compute_data_path):
    """
    load all datafile .csv in folder_path

    Load from files if merging was pre_computed for best performance

    return two pandas.DataFrame
        merged_raw_df_train
        merged_raw_df_test
    """

    raw_df_dict = None
    computed_files = glob.glob(compute_data_path + "/*.csv")
    data_not_precomputed = 'merged_raw_df_train.csv' not in [os.path.basename(fname) for fname in computed_files]

    if data_not_precomputed:
        log.info(f"Loading RAW datasets from path {raw_data_path}")
        datasets_files = glob.glob(raw_data_path + "/*.csv")
        log.info(f"Available datasets files : {datasets_files}")
        raw_df_dict = {}
        if len(datasets_files) > 0:
            for fname in datasets_files:
                try:
                    key = os.path.basename(fname)
                    raw_df_dict[key] = pd.read_csv(fname)
                    log.info(f"loaded file '{key}' : shape{raw_df_dict[key].shape}")
                except Exception as e:
                    log.error(f"Can't load data from file {fname} " + str(e))
        else:
            raise ValueError(f"Can't find datafiles - project can't compute without correct data, please check available files in folder {raw_data_path}.")


    return merge_raw_datasets(compute_data_path, raw_df_dict=raw_df_dict )


def merge_raw_datasets(compute_folder_path, raw_df_dict=None):
    """
    Merge multiple dataframe from dict architecture (ie. load_raw_data() output) based on customer_id
    Load from files if merging was pre_computed for best performance
    return two pandas.DataFrame
        merged_raw_df_train
        merged_raw_df_test
    """

    ## We are using only application_train.csv in this version
    ## But merging strategy can go here
    if raw_df_dict is not None:
        log.info(f"Merging raw datasource")
        merged_raw_df_train = raw_df_dict['application_train.csv']
        merged_raw_df_train.to_csv(f"{compute_folder_path}/merged_raw_df_train.csv", index=False)
        merged_raw_df_test = raw_df_dict['application_test.csv']
        merged_raw_df_test.to_csv(f"{compute_folder_path}/merged_raw_df_test.csv", index=False)

    else:
        log.info(f"Data pre_computed - loading merdeg_raw_df_*  from saved files")
        merged_raw_df_train = pd.read_csv(f"{compute_folder_path}/merged_raw_df_train.csv")
        merged_raw_df_test = pd.read_csv(f"{compute_folder_path}/merged_raw_df_test.csv")

    log.info(f"merged_raw_df_train : shape{merged_raw_df_train.shape}")
    log.info(f"merged_raw_df_test : shape{merged_raw_df_test.shape}")

    return merged_raw_df_train, merged_raw_df_test


def split_train_df(merged_raw_df_train, compute_data_path):
    """
    Split raw_train_df to train and eval sets
    Load from files if split was pre_computed for best performance

    return pandas.DataFrame for train and eval sets
    """

    computed_files = glob.glob(compute_data_path + "/*.csv")
    data_not_precomputed = 'raw_df_train_split.csv' not in [os.path.basename(fname) for fname in computed_files]

    if data_not_precomputed:
        log.info("Spliting merged_raw_df_train into train and eval sets")
        x_train, x_eval, y_train, y_eval = train_test_split(merged_raw_df_train,
                                                            merged_raw_df_train['TARGET'],
                                                            test_size=conf.TEST_SPLIT_RATIO,
                                                            stratify=merged_raw_df_train['TARGET']
                                                            )

        x_train.to_csv(f"{compute_data_path}/raw_df_train_split.csv", index=False)
        x_eval.to_csv(f"{compute_data_path}/raw_df_eval_split.csv", index=False)
    else:
        log.info("Spliting merged_raw_df_train precomputed - loading from saved files")
        x_train = pd.read_csv(f"{compute_data_path}/raw_df_train_split.csv")
        x_eval = pd.read_csv(f"{compute_data_path}/raw_df_eval_split.csv")

    log.info(f"raw_df_train_split : shape{x_train.shape}")
    log.info(f"raw_df_eval_split : shape{x_eval.shape}")

    return x_train, x_eval


def transform_data(raw_df_train_split, raw_df_eval_split, merged_raw_df_test, compute_data_path):
    computed_files = glob.glob(compute_data_path + "/*.csv")
    data_not_precomputed = 'transform_df_train_split.csv' not in [os.path.basename(fname) for fname in computed_files]

    if data_not_precomputed:
        log.info("Fit Transformers")
        ## extract ID and target from datasets
        train_split_id = raw_df_train_split['SK_ID_CURR']
        raw_df_train_split.drop('SK_ID_CURR', inplace=True, axis=1)
        transform_df_train_split_target = raw_df_train_split['TARGET']
        raw_df_train_split.drop('TARGET', inplace=True, axis=1)

        eval_split_id = raw_df_eval_split['SK_ID_CURR']
        raw_df_eval_split.drop('SK_ID_CURR', inplace=True, axis=1)
        transform_df_eval_split_target = raw_df_eval_split['TARGET']
        raw_df_eval_split.drop('TARGET', inplace=True, axis=1)

        test_split_id = merged_raw_df_test['SK_ID_CURR']
        merged_raw_df_test.drop('SK_ID_CURR', inplace=True, axis=1)

        ## transform raw data
        transformer_pipeline = MetaTransformer(verbose=conf.TRANSFORMER_VERBOSITY)
        log.info(f" train shape : {raw_df_train_split.shape}")

        transformer_pipeline.fit(raw_df_train_split, transform_df_train_split_target)

        log.info(f"Pickle transformer pipeline to src/models/transformer_pipeline.pkl")
        with open('src/models/transformer_pipeline.pkl', 'wb') as f:
            pickle.dump(transformer_pipeline, f)

        log.info(f"Transfom and save datasets: train, eval, test")
        transform_df_train_split = transformer_pipeline.transform(raw_df_train_split)
        transform_df_train_split['SK_ID_CURR'] = train_split_id
        transform_df_train_split.to_csv(f"{compute_data_path}/transform_df_train_split.csv", index=False)
        transform_df_train_split_target.to_csv(f"{compute_data_path}/transform_df_train_split_target.csv", index=False)
        
        transform_df_eval_split = transformer_pipeline.transform(raw_df_eval_split)
        transform_df_eval_split['SK_ID_CURR'] = eval_split_id
        transform_df_eval_split.to_csv(f"{compute_data_path}/transform_df_eval_split.csv", index=False)
        transform_df_eval_split_target.to_csv(f"{compute_data_path}/transform_df_eval_split_target.csv", index=False)
        
        transform_df_test = transformer_pipeline.transform(merged_raw_df_test)
        transform_df_test['SK_ID_CURR'] = test_split_id
        transform_df_test.to_csv(f"{compute_data_path}/transform_df_test.csv", index=False)
        
    else:
        log.info(f"Transform data precomputed - loading from saved files")
        transform_df_train_split = pd.read_csv(f"{compute_data_path}/transform_df_train_split.csv")
        transform_df_train_split_target = pd.read_csv(f"{compute_data_path}/transform_df_train_split_target.csv")

        transform_df_eval_split = pd.read_csv(f"{compute_data_path}/transform_df_eval_split.csv")
        transform_df_eval_split_target = pd.read_csv(f"{compute_data_path}/transform_df_eval_split_target.csv")

        transform_df_test = pd.read_csv(f"{compute_data_path}/transform_df_test.csv")

        

    log.info(f" train shape : {transform_df_train_split.shape}")
    log.info(f" eval shape : {transform_df_eval_split.shape}")
    log.info(f" test shape : {transform_df_test.shape}")

    return transform_df_train_split, transform_df_train_split_target, transform_df_eval_split, transform_df_eval_split_target, transform_df_test


def balance_data(transform_df_train_split, transform_df_train_split_target, compute_data_path, method='under'):
    computed_files = glob.glob(compute_data_path + "/*.csv")
    data_not_precomputed = 'undersample_df_train_split.csv' not in [os.path.basename(fname) for fname in computed_files]

    if data_not_precomputed:
        log.info("Balance df_train_split")
        log.info("Make Under_Sampling")
        random_under_sampler = RandomUnderSampler()
        undersample_df_train_split, undersample_df_train_split_target = random_under_sampler.fit_resample(transform_df_train_split, transform_df_train_split_target)
        undersample_df_train_split.to_csv(f"{compute_data_path}/undersample_df_train_split.csv", index=False)
        undersample_df_train_split_target.to_csv(f"{compute_data_path}/undersample_df_train_split_target.csv", index=False)

        log.info("Make Over_Sampling")
        over_sampler = SMOTE()
        overersample_df_train_split, oversample_df_train_split_target = over_sampler.fit_resample( transform_df_train_split, transform_df_train_split_target)
        overersample_df_train_split.to_csv(f"{compute_data_path}/overersample_df_train_split.csv", index=False)
        oversample_df_train_split_target.to_csv(f"{compute_data_path}/oversample_df_train_split_target.csv", index=False)

        if method == 'under':
            df_train_split = undersample_df_train_split
            df_train_split_target = undersample_df_train_split_target
        else:
            df_train_split = overersample_df_train_split
            df_train_split_target = oversample_df_train_split_target

    else:
        log.info("Balance pre-computed - loading data from files")
        if method == 'under':
            df_train_split = pd.read_csv(f"{compute_data_path}/undersample_df_train_split.csv")
            df_train_split_target = pd.read_csv(f"{compute_data_path}/undersample_df_train_split_target.csv")
        else:
            df_train_split = pd.read_csv(f"{compute_data_path}/overersample_df_train_split.csv")
            df_train_split_target = pd.read_csv(f"{compute_data_path}/oversample_df_train_split_target.csv")
    
    log.info(f"balance_df_train_split - shape{df_train_split.shape}")
    log.info(f"balance_df_train_split_target - shape{df_train_split_target.shape}")
    
    return df_train_split, df_train_split_target


def fit_and_save_models(df_train_split, df_train_split_target, df_eval_split, df_eval_split_target):
    computed_files = glob.glob("src/models/*.pkl")
    data_not_precomputed = 'lgbm_undersample.pkl' not in [os.path.basename(fname) for fname in computed_files]

    if data_not_precomputed:
        log.info("Fitting model from undersample dataset")
        params = {
            'num_leaves': [3, 6, 8, 10, 12, 24],
            'min_child_samples' : [100, 200, 300, 500, 800],
            'min_child_weight' : [1e-5, 1e-3, 1e-2, 1e-1, 1, 1e1, 1e2, 1e3, 1e4],
            'subsample' : [0.2, 0.3, 0.4, 0.5, 0.8],
            'colsample_bytree': [0.4, 0.5, 0.6],
            'reg_alpha': [0, 1e-1, 1, 2, 5, 7, 10, 50, 100],
            'reg_lambda': [0, 1e-1, 1, 2, 5, 7, 10, 50, 100]
        }

        params = {'colsample_bytree': 0.9234, 'min_child_samples': 399, 'min_child_weight': 0.1, 'num_leaves': 13, 'reg_alpha': 2, 'reg_lambda': 5, 'subsample': 0.855}

        classifier = lgb.LGBMClassifier(max_depth=1, random_state=314, n_estimators=1000, **params)
        # grid = GridSearchCV(estimator= classifier,
        #                     param_grid= params,
        #                     cv=5,
        #                     n_jobs=4)
        #
        # grid.fit(df_train_split, df_train_split_target)
        classifier.fit(df_train_split, df_train_split_target)
        # lgbm = grid.best_estimator_
        lgbm = classifier
        log.info(f"Pickle classifier to src/models/lgbm_undersample.pkl")
        with open('src/models/lgbm_undersample.pkl', 'wb') as f:
            pickle.dump(lgbm, f)
    else:
        log.info("Model pre-fitted - loading from saved files")
        with open('src/models/lgbm_undersample.pkl', 'rb') as f:
            lgbm = pickle.load(f)

    predicted = lgbm.predict(df_eval_split)
    predicted_proba = lgbm.predict_proba(df_eval_split)

    log.info(f"lgbm score : {lgbm.score(df_eval_split, df_eval_split_target)}")
    log.info(f"lgbm recall {metrics.recall_score(df_eval_split_target, predicted)}")
    log.info(f"lgbm precision {metrics.precision_score(df_eval_split_target, predicted)}")
    log.info(f"lgbm fbeta_2 {metrics.fbeta_score(df_eval_split_target, predicted, beta=2)}")
    log.info(f"lgbm roc_auc {metrics.roc_auc_score(df_eval_split_target, predicted_proba[:,1])}")



if __name__ == '__main__':
    log = logging.getLogger(__name__)
    log.info("Running 'build_project.py' ... ")

    # get directory/file architecture
    current_path = os.getcwd()
    raw_data_path = f"{os.path.join(current_path, *conf.RAW_DATA_PATH.split('/'))}"
    compute_data_path = f"{os.path.join(current_path, *conf.COMPUTE_DATA_PATH.split('/'))}"

    # load raw dataset
    # merge raw dataset
    # save merged data as .csv
    merged_raw_df_train, merged_raw_df_test = load_raw_data(raw_data_path, compute_data_path)

    # generate raw splits (train + eval) from train merged_data + save splits as .csv
    raw_df_train_split, raw_df_eval_split = split_train_df(merged_raw_df_train, compute_data_path)


    # fit transformers on train split + apply transformers on eval and test data + save computed datasets as .csv + save transformers as pickle
    transform_df_train_split, transform_df_train_split_target, transform_df_eval_split, transform_df_eval_split_target, transform_df_test = transform_data(raw_df_train_split, raw_df_eval_split, merged_raw_df_test, compute_data_path)

    # generate balanced train_datasets + save as .csv
    balanced_df_train_split, balanced_df_train_split_target = balance_data(transform_df_train_split, transform_df_train_split_target, compute_data_path, method='over')
    
    
    # fit model on balanced datasets + save bests models as pickle
    fit_and_save_models(balanced_df_train_split, balanced_df_train_split_target, transform_df_eval_split, transform_df_eval_split_target)
