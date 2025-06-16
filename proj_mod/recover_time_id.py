#This file is based on work done in https://www.kaggle.com/competitions/optiver-realized-volatility-prediction/discussion/274970. 

import glob

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.manifold import TSNE
from sklearn.preprocessing import minmax_scale


def calc_price_from_tick(df):
    tick = sorted(np.diff(sorted(np.unique(df.values.flatten()))))[0]
    return 0.01 / tick


def calc_prices(r):
    df = pd.read_parquet(r.book_path,
                         columns=[
                             'time_id',
                             'ask_price1',
                             'ask_price2',
                             'bid_price1',
                             'bid_price2'
                         ])
    df = df.groupby('time_id') \
        .apply(calc_price_from_tick, include_groups=False).to_frame('price').reset_index()
    df['stock_id'] = r.stock_id
    return df

def reconstruct_time_id_order(str_path):
    """
    A function that takes the path to book_train.parquet and return a list of time id in recovered order. The function is based on work done by https://www.kaggle.com/competitions/optiver-realized-volatility-prediction/discussion/274970. 
    
    :param str_path: The path to book_train.parquet. 
    :return: A list of time id in recover order. 
    """
    # Added 'sorted' for consistent ordering between computers
    paths = sorted(glob.glob(str_path+'/**/*.parquet'))

    df_files = pd.DataFrame(
        {'book_path': paths}) \
        .eval('stock_id = book_path.str.extract("stock_id=(\d+)").astype("int")',
              engine='python')

    # build price matrix using tick-size
    # Changed 'n_jobs' from 4 to -1 (uses all processors except for one)
    df_prices = pd.concat(
        Parallel(n_jobs=-1)(
            delayed(calc_prices)(r) for _, r in df_files.iterrows()
        )
    )
    df_prices = df_prices.pivot(index='time_id', columns='stock_id', values='price')

    # t-SNE to recovering time-id order
    # 'n_iter' was replaced by 'max_iter' in sklearn >= 1.5
    clf = TSNE(
        n_components=1,
        perplexity=400,
        random_state=0,
        max_iter=2000
    )
    compressed = clf.fit_transform(
        pd.DataFrame(minmax_scale(df_prices.fillna(df_prices.mean())))
    )

    order = np.argsort(compressed[:, 0])
    ordered=df_prices.set_index(order)

    # correct direction of time-id order using known stock (id61 = AMZN)
    if ordered[61].iloc[0] > ordered[61].iloc[-1]:
        ordered = ordered.reindex(ordered.index[::-1])\
            .reset_index(drop=True)
            
    time_order = ordered.index

    return [int(df_prices.index[id]) for id in time_order]