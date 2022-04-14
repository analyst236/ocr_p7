import pandas as pd
import settings as conf
import logging
import logging.config
import os
import glob

from sklearn.model_selection import train_test_split

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



    # generate balanced train_datasets + save as .csv

    # fit model on balanced datasets + save bests models as pickle

    pass
