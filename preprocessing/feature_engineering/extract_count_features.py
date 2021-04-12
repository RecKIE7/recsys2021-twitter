import sys
sys.path.append('../..')


import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
import pandas as pd, numpy as np, gc
from tqdm import tqdm
from sklearn.metrics import precision_recall_curve, auc, log_loss
import subprocess
import dask.multiprocessing
import joblib
from numba import jit, njit, prange

tqdm.pandas()
pd.set_option('display.max_columns', 500)
pd.set_option('display.max_rows', 500)

import core.config as conf


class ExtractCountFeatures(object):
    def __init__(self, file):
        
        def compute_prauc(gt, pred, nafill=True):
            if nafill:
                pred[ np.isnan(pred) ] = np.nanmean( pred )
            prec, recall, thresh = precision_recall_curve(gt, pred)
            prauc = auc(recall, prec)
            return prauc

        @jit
        def fast_auc(y_true, y_prob):
            y_true = np.asarray(y_true)
            y_true = y_true[np.argsort(y_prob)]
            nfalse = 0
            auc = 0
            n = len(y_true)
            for i in range(n):
                y_i = y_true[i]
                nfalse += (1 - y_i)
                auc += y_i * nfalse
            auc /= (nfalse * (n - nfalse))
            return auc

        @njit
        def numba_log_loss(y,x):
            n = x.shape[0]
            ll = 0.
            for i in prange(n):
                if y[i]<=0.:
                    ll += np.log(1-x[i] + 1e-15 )
                else:
                    ll += np.log(x[i] + 1e-15)
            return -ll / n

        def compute_rce(gt , pred, nafill=True, verbose=0):
            if nafill:
                pred[ np.isnan(pred) ] = np.nanmean( pred )
                
            cross_entropy = numba_log_loss( gt, pred  )
            
            yt = np.mean(gt>0)     
            strawman_cross_entropy = -(yt*np.log(yt) + (1 - yt)*np.log(1 - yt))
            
            if verbose:
                print( "logloss: {0:.5f} / {1:.5f} = {2:.5f}".format(cross_entropy, strawman_cross_entropy, cross_entropy/strawman_cross_entropy))
                print( 'mean:    {0:.5f} / {1:.5f}'.format( np.nanmean( pred ) , yt  ) )
            
            return (1.0 - cross_entropy/strawman_cross_entropy)*100.0

        def save_memory( df ):
            features = df.columns
            for i in range( df.shape[1] ):
                if df.dtypes[i] == 'uint8':
                    df[features[i]] = df[features[i]].astype( np.int8 )
                    gc.collect()
                elif df.dtypes[i] == 'bool':
                    df[features[i]] = df[features[i]].astype( np.int8 )
                    gc.collect()
                elif df.dtypes[i] == 'uint32':
                    df[features[i]] = df[features[i]].astype( np.int32 )
                    gc.collect()
                elif df.dtypes[i] == 'int64':
                    df[features[i]] = df[features[i]].astype( np.int32 )
                    gc.collect()
                elif df.dtypes[i] == 'float64':
                    df[features[i]] = df[features[i]].astype( np.float32 )
                    gc.collect()

        



if __name__ == '__main__':
    from utils.cuda_cluster import *
    from utils.dataset import read_data, factorize_small_cardinality

    fire.Fire(ExtractCountFeatures)
