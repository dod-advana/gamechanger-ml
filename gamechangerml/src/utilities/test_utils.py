import os
import pandas as pd
from dateutil import parser
from gamechangerml.api.utils.logger import logger
from gamechangerml.configs.config import ValidationConfig

MATAMO_DIR = ValidationConfig.DATA_ARGS['matamo_dir']
SEARCH_HIST = ValidationConfig.DATA_ARGS['search_hist_dir']

def convert_timestamp_to_datetime(timestamp):
    return pd.to_datetime(parser.parse(timestamp).strftime("%Y-%m-%d"))

## filter users and dates when csv read in
def filter_date_range(df, start_date, end_date):
    if 'createdAt' in df.columns:
        timecol = 'createdAt'
    elif 'searchtime' in df.columns:
        timecol = 'searchtime'
    df['dt'] = df[timecol].apply(lambda x: convert_timestamp_to_datetime(x))
    logger.info(f"Available date range: {str(min(df['dt']))} - {str(max(df['dt']))}")
    subset = df.copy()
    if start_date:
        subset = subset[subset['dt'] >= pd.to_datetime(start_date)]
    if end_date:
        subset = subset[subset['dt'] <= pd.to_datetime(end_date)]
    logger.info(f"New date range: {str(min(subset['dt']))} - {str(max(subset['dt']))}")
    return subset

def concat_csvs(directory):
    '''Combines csvs in directory into one df; drops entirly null columns'''
    df = pd.DataFrame()
    logger.info(str(directory))
    csvs = [i for i in os.listdir(directory) if i.split('.')[-1]=='csv']
    logger.info(f"CSVs: {str(csvs)}")
    for i in csvs:
        f = pd.read_csv(os.path.join(directory, i))
        df = pd.concat([df, f])
    return df

def concat_matamo():
    return concat_csvs(MATAMO_DIR)

def concat_search_hist():
    return concat_csvs(SEARCH_HIST)

def get_most_recent_dir(parent_dir):

    subdirs = [os.path.join(parent_dir, d) for d in os.listdir(parent_dir) if os.path.isdir(os.path.join(parent_dir, d))]
    return max(subdirs, key=os.path.getctime)