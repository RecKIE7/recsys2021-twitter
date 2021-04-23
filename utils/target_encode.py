import numpy as np
class MTE_one_shot:
    
    def __init__(self, folds, smooth, seed=42):
        self.folds = folds
        self.seed = seed
        self.smooth = smooth
        
    def fit_transform(self, train, x_col, y_col, y_mean=None, out_col = None, out_dtype=None):
        
        self.y_col = y_col
        np.random.seed(self.seed)
        
        if 'fold' not in train.columns:
            fsize = len(train)//self.folds
            train['fold'] = 1
            train['fold'] = train['fold'].cumsum()
            train['fold'] = train['fold']//fsize
            train['fold'] = train['fold']%self.folds
        
        if out_col is None:
            tag = x_col if isinstance(x_col,str) else '_'.join(x_col)
            out_col = f'TE_{tag}_{self.y_col}'
        
        if y_mean is None:
            y_mean = train[y_col].mean()#.compute().astype('float32')
        self.mean = y_mean
        
        cols = ['fold',x_col] if isinstance(x_col,str) else ['fold']+x_col
        
        agg_each_fold = train.groupby(cols).agg({y_col:['count','sum']}).reset_index()
        agg_each_fold.columns = cols + ['count_y','sum_y']
        
        agg_all = agg_each_fold.groupby(x_col).agg({'count_y':'sum','sum_y':'sum'}).reset_index()
        cols = [x_col] if isinstance(x_col,str) else x_col
        agg_all.columns = cols + ['count_y_all','sum_y_all']
        
        agg_each_fold = agg_each_fold.merge(agg_all,on=x_col,how='left')
        agg_each_fold['count_y_all'] = agg_each_fold['count_y_all'] - agg_each_fold['count_y']
        agg_each_fold['sum_y_all'] = agg_each_fold['sum_y_all'] - agg_each_fold['sum_y']
        agg_each_fold[out_col] = (agg_each_fold['sum_y_all']+self.smooth*self.mean)/(agg_each_fold['count_y_all']+self.smooth)
        agg_each_fold = agg_each_fold.drop(['count_y_all','count_y','sum_y_all','sum_y'],axis=1)
        
        agg_all[out_col] = (agg_all['sum_y_all']+self.smooth*self.mean)/(agg_all['count_y_all']+self.smooth)
        agg_all = agg_all.drop(['count_y_all','sum_y_all'],axis=1)
        self.agg_all = agg_all
        np.save('agg_all.npy', agg_all)
        
        train.columns
        cols = ['fold',x_col] if isinstance(x_col,str) else ['fold']+x_col
        train = train.merge(agg_each_fold,on=cols,how='left')
        del agg_each_fold
        #self.agg_each_fold = agg_each_fold
        #train[out_col] = train.map_partitions(lambda cudf_df: cudf_df[out_col].nans_to_nulls())
        train[out_col] = train[out_col].fillna(self.mean)
        
        if out_dtype is not None:
            train[out_col] = train[out_col].astype(out_dtype)
        return train
    
    def transform(self, test, x_col, out_col = None, out_dtype=None):
        self.agg_all = np.load('agg_all.npy')
        if out_col is None:
            tag = x_col if isinstance(x_col,str) else '_'.join(x_col)
            out_col = f'TE_{tag}_{self.y_col}'
        test = test.merge(self.agg_all,on=x_col,how='left')
        test[out_col] = test[out_col].fillna(self.mean)
        if out_dtype is not None:
            test[out_col] = test[out_col].astype(out_dtype)
        return test