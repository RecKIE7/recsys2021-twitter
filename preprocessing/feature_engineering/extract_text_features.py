import sys
sys.path.append('../..')

import numpy as np
import pandas as pd
from tqdm import tqdm
import gc
import glob
import os.path
import hashlib


import core.config as conf


class ExtractTextFeatures(object):
    def __init__(self, file):
        tqdm.pandas()

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

        def extract_hash(text, split_text='@', no=0):
            text = text.lower()
            uhash = ''
            text_split = text.split('@')
            if len(text_split)>(no+1):
                text_split = text_split[no+1].split(' ')
                cl_loop = True
                uhash += clean_text(text_split[0])
                while cl_loop:
                    if len(text_split)>1:
                        if text_split[1] in ['_']:
                            uhash += clean_text(text_split[1]) + clean_text(text_split[2])
                            text_split = text_split[2:]
                        else:
                            cl_loop = False
                    else:
                        cl_loop = False
            hash_object = hashlib.md5(uhash.encode('utf-8'))
            return hash_object.hexdigest()

        def clean_text(text):
            if len(text)>1:
                if text[-1] in ['!', '?', ':', ';', '.', ',']:
                    return(text[:-1])
            return(text)

        
        path = f'{conf.preproc_path}/step3_output/{file}'
        train = pd.read_parquet(f'{path}/train-tweet-1.parquet',  engine='pyarrow')


        WORDS = {}
        DF = []
        for tweet in tqdm(train['tweet'].unique()):
            words = tweet.split(' ')
            for w in words:
                if w not in WORDS:
                    WORDS[w] = 1
                else:
                    WORDS[w]+= 1
        gc.collect()
                        
        count=0
        for w in tqdm(WORDS):
            WORDS[w] = [ WORDS[w], count ]
            count+=1
        gc.collect()


        def freq_encode_words( vs ):
            li=[]
            lf=[]
            for v in vs.split(' '):
                if v not in ['','[',']','.','!','@','_','#']:
                    f,i = WORDS[v]
                    if f<100000:
                        if f>2:
                            li.append( str(i) )
                            #li.append( v )
                            lf.append( f )
            return ' '.join( list((np.array(li)[np.argsort(lf)] )) )    

        def ret_word( x, rw=0 ):
            x = x.split(' ')
            if rw==0:
                if len(x)>=1:
                    return x[0]
            elif rw==1:
                if len(x)>=2:
                    return x[1]
            elif rw== -1:
                if len(x)>=1:
                    return x[-1]
            elif rw== -2:
                if len(x)>=2:
                    return x[-2]

            return '-1'

        
        DF = []
        train['tweet_nortsign'] = train['tweet'].str.replace('\[CLS\] RT @', '')
        train['count_words']    = train['tweet'].str.count(' ')
        train['count_char']     = train['tweet'].progress_apply(lambda x: len(x))
        train['count_ats']      = train['tweet_nortsign'].str.count('@')
        train['hash0']          = train['tweet_nortsign'].progress_apply(lambda x: extract_hash(x))
        train['hash1']          = train['tweet_nortsign'].progress_apply(lambda x: extract_hash(x, no=1))
        train['tw_uhash']       = train['tweet'].progress_apply(lambda x: extract_hash(x, split_text='RT @', no=0))
        train['tw_hash']        = train['tweet'].progress_apply(lambda x: hash(x)%1000000000 )

        train['tweet']          = train['tweet'].progress_apply(lambda x: freq_encode_words(x) )
        train['tw_freq_hash']   = train['tweet'].progress_apply(lambda x: hash(x)%1000000000 )
        train['tw_first_word']  = train['tweet'].progress_apply(lambda x: ret_word(x,0) )
        train['tw_second_word'] = train['tweet'].progress_apply(lambda x: ret_word(x,1) )
        train['tw_last_word']   = train['tweet'].progress_apply(lambda x: ret_word(x,-1) )
        train['tw_llast_word']  = train['tweet'].progress_apply(lambda x: ret_word(x,-2) )
        train['tw_len']         = train['tweet'].progress_apply(lambda x: len(x.split(' ')) )


        DF = train[['id','count_ats', 'count_char', 'count_words', 'hash0', 'hash1', 'tw_uhash','tw_hash','tw_freq_hash','tw_first_word','tw_second_word','tw_last_word','tw_llast_word','tw_len']]
        del train
        gc.collect()

        save_memory( DF )
        DF = DF.reset_index( drop=True )
        gc.collect()

        uhashes = pd.concat([DF['hash0'], DF['hash1'], DF['tw_uhash']], axis=0)
        gc.collect()
        uhashes = uhashes.value_counts()
        gc.collect()
        uhashes = uhashes.reset_index().reset_index()
        gc.collect()
        uhashes['uid'] = np.arange(0,uhashes.shape[0] )
        uhashes.head()

        DF['tw_hash0']    = pd.merge( DF[['hash0']]  , uhashes[['index','uid']], left_on='hash0'  , right_on='index', how='left' )['uid']
        gc.collect()
        DF['tw_hash1']    = pd.merge( DF[['hash1']]  , uhashes[['index','uid']], left_on='hash1'  , right_on='index', how='left' )['uid']
        gc.collect()
        DF['tw_rt_uhash'] = pd.merge( DF[['tw_uhash']], uhashes[['index','uid']], left_on='tw_uhash', right_on='index', how='left' )['uid']
        gc.collect()

        del DF['hash0']
        del DF['hash1']
        del DF['tw_uhash']
        gc.collect()
        save_memory( DF )

        DF['tw_hash']        = pd.factorize( DF['tw_hash'] )[0]
        DF['tw_freq_hash']   = pd.factorize( DF['tw_freq_hash'] )[0]
        DF['tw_first_word']  = pd.factorize( DF['tw_first_word'] )[0]
        DF['tw_second_word'] = pd.factorize( DF['tw_second_word'] )[0]
        DF['tw_last_word']   = pd.factorize( DF['tw_last_word'] )[0]
        DF['tw_llast_word']  = pd.factorize( DF['tw_llast_word'] )[0]
        gc.collect()
        
        DF['tw_hash']        = DF['tw_hash'].astype(np.int32)
        DF['tw_freq_hash']   = DF['tw_freq_hash'].astype(np.int32)
        DF['tw_first_word']  = DF['tw_first_word'].astype(np.int32)
        DF['tw_second_word'] = DF['tw_second_word'].astype(np.int32)
        DF['tw_last_word']   = DF['tw_last_word'].astype(np.int32)
        DF['tw_llast_word']  = DF['tw_llast_word'].astype(np.int32)
        gc.collect()


        path = f'{conf.preproc_path}/step4_output/{file}'
        DF.to_parquet( f'{path}/text-processings-1.parquet' )
        gc.collect()
        del DF


if __name__ == '__main__':
    fire.Fire(ExtractTextFeatures)
