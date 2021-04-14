from itertools import chain
import joblib
from numba import jit, njit, prange
import pandas as pd, numpy as np, gc

def chainer(s, sep='\t'):
    return list(chain.from_iterable(s.str.split(sep)))

def plot_graphs(history, string):
    plt.plot(history.history[string])
    plt.plot(history.history['val_'+string], '')
    plt.xlabel("Epochs")
    plt.ylabel(string)
    plt.legend([string, 'val_'+string])
    plt.show()

def save_memory(df):
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

def split_join(ds, sep):
    return ds.str.replace('\t', '_')