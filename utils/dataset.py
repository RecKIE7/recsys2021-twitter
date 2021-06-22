import sys, os
sys.path.append('..')

from tqdm import tqdm
import pickle

from collections import defaultdict
from utils.target_encode import MTE_one_shot
from utils.preprocessing import *
import core.config as conf
from transformers import *

class Dataset:
    def __init__(self, train=False, target_encoding=conf.target_encoding):
        self.all_features_to_idx = dict(zip(conf.raw_features, range(len(conf.raw_features))))
        self.train = train
        self.target_encoding = target_encoding
        self.default_values = conf.default_values

    def preprocess(self, df, TARGET_id=conf.LIKE):
        # target = conf.target[TARGET_id]
        df = self.set_dataframe_types(df)
        # df = df.drop('text_tokens', axis=1)
        df = feature_extraction(df, features=conf.used_features, train=self.train) 
        if conf.target_encoding == 1:    
            for c in ([
                ['engager_id'],
                ['engager_id','tweet_type','language'],
                ['creator_id'],
                ['domains','media','tweet_type','language']
                ]):
                fname = 'TE_'+'_'.join(c)+'_'+target
                print( fname )
                df[fname] = tartget_encoding( df, c, target, 20, 0 )

        elif conf.target_encoding == 2:
            for c in tqdm([
                ['engager_id'],
                ['engager_id','tweet_type','language'],
                ['creator_id'],
                ['domains','media','tweet_type','language']
                ]):
                
                out_col = 'TE_'+'_'.join(c)+'_'+target
                encoder_path = f'{out_col}.pkl'
                
                if os.path.exists(encoder_path):
                    with open(encoder_path, 'rb') as f:
                        encoder = pickle.load(f)
                else:
                    encoder = MTE_one_shot(folds=5,smooth=20)

                if self.train:
                    
                    df = encoder.fit_transform(df, c, target, out_col=out_col, out_dtype='float32')
                    with open(encoder_path, 'wb') as f:
                        pickle.dump(encoder, f)
                else:
                    df = encoder.transform(df, c, out_col=out_col, out_dtype='float32')
        
            del encoder
            
        elif conf.target_encoding == 3:
            df['creator_account_creation_group'] = df['creator_account_creation'].apply(grouping, args=(1136041200, 31536000))
            df['creator_follower_count_group'] = df['creator_follower_count'].apply(grouping, args=(0, 0, [10, 100, 1000, 100000]))

            for c in tqdm([
                ['creator_account_creation_group'],
                ['creator_follower_count_group'],
                ['creator_account_creation_group', 'creator_follower_count_group'],
                ]):
                
                out_col = 'TE_'+'_'.join(c)+'_'+target
                encoder_path = f'{out_col}.pkl'
                
                if os.path.exists(encoder_path):
                    with open(encoder_path, 'rb') as f:
                        encoder = pickle.load(f)
                else:
                    encoder = MTE_one_shot(folds=5,smooth=20)

                if self.train:
                    df = encoder.fit_transform(df, c, target, out_col=out_col, out_dtype='float32')
                    with open(encoder_path, 'wb') as f:
                        pickle.dump(encoder, f)
                else:
                    df = encoder.transform(df, c, out_col=out_col, out_dtype='float32')
        
            del encoder

        return df
    
    def raw_preprocess(self, df, TARGET_id=conf.LIKE) : 
        df = self.set_dataframe_types(df)
        labels = ['reply_timestamp', 'retweet_timestamp', 'comment_timestamp', 'like_timestamp']
        df['reply'] = df['reply_timestamp'].apply(lambda x: 1 if x > 0 else 0).astype(np.int32)
        df['retweet'] = df['retweet_timestamp'].apply(lambda x: 1 if x > 0 else 0).astype(np.int32)
        df['comment'] = df['comment_timestamp'].apply(lambda x: 1 if x > 0 else 0).astype(np.int32)
        df['like'] = df['like_timestamp'].apply(lambda x: 1 if x > 0 else 0).astype(np.int32) 
        df = df.drop(labels, axis=1)
       
        return df
    
    def set_dataframe_types(self, df):
        df['id']   = np.arange( df.shape[0] )
        df['id']   = df['id'].astype(np.uint32)

        if self.train:
            df['reply_timestamp']   = df['reply_timestamp'].fillna(0)
            df['retweet_timestamp'] = df['retweet_timestamp'].fillna(0)
            df['comment_timestamp'] = df['comment_timestamp'].fillna(0)
            df['like_timestamp']    = df['like_timestamp'].fillna(0)

            df['reply_timestamp']   = df['reply_timestamp'].astype(np.uint32)
            df['retweet_timestamp'] = df['retweet_timestamp'].astype(np.uint32)
            df['comment_timestamp'] = df['comment_timestamp'].astype(np.uint32)
            df['like_timestamp']    = df['like_timestamp'].astype(np.uint32)
        
        df['tweet_timestamp']         = df['tweet_timestamp'].astype( np.uint32 )
        df['creator_follower_count']  = df['creator_follower_count'].astype( np.uint32 )
        df['creator_following_count'] = df['creator_following_count'].astype( np.uint32 )
        df['creator_account_creation']= df['creator_account_creation'].astype( np.uint32 )
        df['engager_follower_count'] = df['engager_follower_count'].fillna(0)
        df['engager_follower_count']  = df['engager_follower_count'].astype( np.uint32 )
        df['engager_following_count'] = df['engager_following_count'].fillna(0)
        df['engager_following_count'] = df['engager_following_count'].astype( np.uint32 )
        df['engager_account_creation'] = df['engager_account_creation'].fillna(0)
        df['engager_account_creation']= df['engager_account_creation'].astype( np.uint32 )

        return df
    
    def cal_valid_pickle(self, df) :
        ## main language
        lang_dict_path = conf.pickle_data + 'language_dict.pkl'
        pred_pickle_path = conf.pred_pickle_path + "user_main_language.pkl"

        user_language = defaultdict(list)

        for index, row in df.iterrows():
            creator_id = row['creator_id']
            language = row['language']
            user_language[creator_id].append(language)

        for user in user_language :
            max_lang = max(user_language[user])
            user_language[user] = max_lang


        with open(pred_pickle_path, 'wb') as f:
            pickle.dump(user_language, f)

        del df, user_language
    
    def user_engagements(self, df, train=True):
                
        pickle_path = conf.pickle_data
        pred_pickle_path = conf.pred_pickle_path

        '''
        ############ language mapping ############
        '''

        language_dict_path = pickle_path + "language_dict.pkl"

        if os.path.exists(language_dict_path ) :
            with open(language_dict_path , 'rb') as f :
                language_dict = pickle.load(f)
                language_dict = defaultdict(lambda :66, language_dict)
        
        df['language'] = df.apply(lambda x : language_dict[x['language']], axis = 1)
        
        
        # main_language

        user_main_language_path = pickle_path + "user_main_language.pkl"
        pred_main_language_path = pred_pickle_path + "user_main_language.pkl"

        if os.path.exists(user_main_language_path) :
            with open(user_main_language_path, 'rb') as f :
                user_main_language = pickle.load(f)

        if train == False :
            self.cal_valid_pickle(df)
            if os.path.exists(pred_main_language_path) :
                with open(pred_main_language_path, 'rb') as f :
                    pred_main_language = pickle.load(f)

            for user in pred_main_language :
                if user not in user_main_language :
                    user_main_language[user] = pred_main_language[user]
            del pred_main_language

        user_main_language = defaultdict(lambda : 0, user_main_language)

        df['creator_main_language'] = df['creator_id'].map(user_main_language)
        df['engager_main_language'] = df['engager_id'].map(user_main_language)
        df['creator_main_language'] = df['creator_main_language'].astype(np.int32) 
        df['engager_main_language'] = df['engager_main_language'].astype(np.int32)
        df['creator_and_engager_have_same_main_language'] = df['creator_main_language'] == df['engager_main_language']
        df['is_tweet_in_creator_main_language'] = df['creator_main_language'] == df['language']
        df['is_tweet_in_engager_main_language'] = df['engager_main_language'] == df['language']
        df['creator_and_engager_have_same_main_language'] = df['creator_and_engager_have_same_main_language'].astype(np.int)
        df['is_tweet_in_creator_main_language'] = df['is_tweet_in_creator_main_language'].astype(np.int)
        df['is_tweet_in_engager_main_language'] = df['is_tweet_in_engager_main_language'].astype(np.int)


        '''
        ############ engager mapping ############
        '''

        engagement_like_path = pickle_path + "engager-engagement-like.pkl"
        if os.path.exists(engagement_like_path ) :
            with open(engagement_like_path , 'rb') as f :
                engagement_like = pickle.load(f)
                engagement_like = defaultdict(lambda : self.default_values['engager_feature_number_of_previous_like_engagement'], engagement_like)

        df['engager_feature_number_of_previous_like_engagement'] = df.apply(lambda x : engagement_like[x['engager_id']], axis = 1)
        del engagement_like
        
        engagement_reply_path = pickle_path + "engager-engagement-reply.pkl"
        if os.path.exists(engagement_reply_path ) :
            with open(engagement_reply_path , 'rb') as f :
                engagement_reply = pickle.load(f)
                engagement_reply = defaultdict(lambda : self.default_values['engager_feature_number_of_previous_reply_engagement'], engagement_reply)


        df['engager_feature_number_of_previous_reply_engagement'] = df.apply(lambda x : engagement_reply[x['engager_id']], axis = 1)
        del engagement_reply

        engagement_retweet_path = pickle_path + "engager-engagement-retweet.pkl"
        if os.path.exists(engagement_retweet_path ) :
            with open(engagement_retweet_path , 'rb') as f :
                engagement_retweet = pickle.load(f)
                engagement_retweet = defaultdict(lambda : self.default_values['engager_feature_number_of_previous_retweet_engagement'], engagement_retweet)


        df['engager_feature_number_of_previous_retweet_engagement'] = df.apply(lambda x : engagement_retweet[x['engager_id']], axis = 1)
        del engagement_retweet

        engagement_comment_path = pickle_path + "engager-engagement-comment.pkl"
        if os.path.exists(engagement_comment_path ) :
            with open(engagement_comment_path , 'rb') as f :
                engagement_comment = pickle.load(f)
                engagement_comment = defaultdict(lambda : self.default_values['engager_feature_number_of_previous_comment_engagement'], engagement_comment)


        df['engager_feature_number_of_previous_comment_engagement'] = df.apply(lambda x : engagement_comment[x['engager_id']], axis = 1)
        del engagement_comment

        df['number_of_engagements_positive'] = df.apply(lambda x : x['engager_feature_number_of_previous_like_engagement'] + x['engager_feature_number_of_previous_retweet_engagement'] + x['engager_feature_number_of_previous_reply_engagement'] + x['engager_feature_number_of_previous_comment_engagement'], axis = 1)
        
        '''
        if target != "reply":
            df = df.drop('engager_feature_number_of_previous_reply_engagement', axis = 1)
            
        if target != "like":
            df = df.drop('engager_feature_number_of_previous_like_engagement', axis = 1)

        if target != "retweet":
            df = df.drop('engager_feature_number_of_previous_retweet_engagement', axis = 1)

        if target != "comment":
            df = df.drop('engager_feature_number_of_previous_comment_engagement', axis = 1)
        '''
        
        df['number_of_engagements_ratio_reply'] = df.apply(lambda x : x['engager_feature_number_of_previous_reply_engagement'] / x['number_of_engagements_positive'] if x['number_of_engagements_positive'] != 0 else 0, axis = 1)
        df['number_of_engagements_ratio_like'] = df.apply(lambda x : x['engager_feature_number_of_previous_like_engagement'] / x['number_of_engagements_positive'] if x['number_of_engagements_positive'] != 0 else 0, axis = 1)
        df['number_of_engagements_ratio_retweet'] = df.apply(lambda x :  x['engager_feature_number_of_previous_retweet_engagement'] / x['number_of_engagements_positive'] if x['number_of_engagements_positive'] != 0 else 0, axis = 1)
        df['number_of_engagements_ratio_comment'] = df.apply(lambda x : x['engager_feature_number_of_previous_comment_engagement'] / x['number_of_engagements_positive'] if x['number_of_engagements_positive'] != 0 else 0, axis = 1)


        '''
        ############ creator mapping ############
        '''

        engagement_like_path = pickle_path + "creator-engagement-like.pkl"
        if os.path.exists(engagement_like_path ) :
            with open(engagement_like_path , 'rb') as f :
                engagement_like = pickle.load(f)
                engagement_like = defaultdict(lambda : self.default_values['creator_feature_number_of_previous_like_engagement'], engagement_like)

        df['creator_feature_number_of_previous_like_engagement'] = df.apply(lambda x : engagement_like[x['creator_id']], axis = 1)
        del engagement_like
        
        engagement_reply_path = pickle_path + "creator-engagement-reply.pkl"
        if os.path.exists(engagement_reply_path ) :
            with open(engagement_reply_path , 'rb') as f :
                engagement_reply = pickle.load(f)
                engagement_reply = defaultdict(lambda : self.default_values['creator_feature_number_of_previous_reply_engagement'], engagement_reply)


        df['creator_feature_number_of_previous_reply_engagement'] = df.apply(lambda x : engagement_reply[x['creator_id']], axis = 1)
        del engagement_reply

        engagement_retweet_path = pickle_path + "creator-engagement-retweet.pkl"
        if os.path.exists(engagement_retweet_path ) :
            with open(engagement_retweet_path , 'rb') as f :
                engagement_retweet = pickle.load(f)
                engagement_retweet = defaultdict(lambda : self.default_values['creator_feature_number_of_previous_retweet_engagement'], engagement_retweet)


        df['creator_feature_number_of_previous_retweet_engagement'] = df.apply(lambda x : engagement_retweet[x['creator_id']], axis = 1)
        del engagement_retweet

        engagement_comment_path = pickle_path + "creator-engagement-comment.pkl"
        if os.path.exists(engagement_comment_path ) :
            with open(engagement_comment_path , 'rb') as f :
                engagement_comment = pickle.load(f)
                engagement_comment = defaultdict(lambda : self.default_values['creator_feature_number_of_previous_comment_engagement'], engagement_comment)


        df['creator_feature_number_of_previous_comment_engagement'] = df.apply(lambda x : engagement_comment[x['creator_id']], axis = 1)
        del engagement_comment

        df['creator_number_of_engagements_positive'] = df.apply(lambda x : x['creator_feature_number_of_previous_like_engagement'] + x['creator_feature_number_of_previous_retweet_engagement'] + x['creator_feature_number_of_previous_reply_engagement'] + x['creator_feature_number_of_previous_comment_engagement'], axis = 1)
        
        '''
        if target != "reply":
            df = df.drop('engager_feature_number_of_previous_reply_engagement', axis = 1)
            
        if target != "like":
            df = df.drop('engager_feature_number_of_previous_like_engagement', axis = 1)

        if target != "retweet":
            df = df.drop('engager_feature_number_of_previous_retweet_engagement', axis = 1)

        if target != "comment":
            df = df.drop('engager_feature_number_of_previous_comment_engagement', axis = 1)
        '''
        
        df['creator_number_of_engagements_ratio_reply'] = df.apply(lambda x : x['creator_feature_number_of_previous_reply_engagement'] / x['creator_number_of_engagements_positive'] if x['creator_number_of_engagements_positive'] != 0 else 0, axis = 1)
        df['creator_number_of_engagements_ratio_like'] = df.apply(lambda x : x['creator_feature_number_of_previous_like_engagement'] / x['creator_number_of_engagements_positive'] if x['creator_number_of_engagements_positive'] != 0 else 0, axis = 1)
        df['creator_number_of_engagements_ratio_retweet'] = df.apply(lambda x : x['creator_feature_number_of_previous_retweet_engagement'] / x['creator_number_of_engagements_positive'] if x['creator_number_of_engagements_positive'] != 0 else 0, axis = 1)
        df['creator_number_of_engagements_ratio_comment'] = df.apply(lambda x : x['creator_feature_number_of_previous_comment_engagement'] / x['creator_number_of_engagements_positive'] if x['creator_number_of_engagements_positive'] != 0 else 0, axis = 1)

        return df

    def tweet_engagements(self, df):
        pickle_path = conf.pickle_data

        # all engagement
        ### 1 ###
        tweet_engagement_path = pickle_path + "tweet_id_engagement_new_1.pkl"

        if os.path.exists(tweet_engagement_path ) :
            with open(tweet_engagement_path, 'rb') as f :
                tweet_engagements = pickle.load(f)
                tweet_engagements = defaultdict(lambda : -1, tweet_engagements)

        df['number_of_tweet_engagements'] = df.apply(lambda x : tweet_engagements[x['tweet_id']], axis = 1)

        df1 = df[df['number_of_tweet_engagements'] != -1]
        df = df[df['number_of_tweet_engagements'] == -1]

        ### 2 ###
        tweet_engagement_path = pickle_path + "tweet_id_engagement_new_2.pkl"

        if os.path.exists(tweet_engagement_path ) :
            with open(tweet_engagement_path, 'rb') as f :
                tweet_engagements = pickle.load(f)
                tweet_engagements = defaultdict(lambda : -1, tweet_engagements)

        df['number_of_tweet_engagements'] = df.apply(lambda x : tweet_engagements[x['tweet_id']], axis = 1)

        df2 = df[df['number_of_tweet_engagements'] != -1]
        df = df[df['number_of_tweet_engagements'] == -1]

        ### 3 ###
        tweet_engagement_path = pickle_path + "tweet_id_engagement_new_3.pkl"

        if os.path.exists(tweet_engagement_path ) :
            with open(tweet_engagement_path, 'rb') as f :
                tweet_engagements = pickle.load(f)
                tweet_engagements = defaultdict(lambda : -1, tweet_engagements)

        df['number_of_tweet_engagements'] = df.apply(lambda x : tweet_engagements[x['tweet_id']], axis = 1)

        df3 = df[df['number_of_tweet_engagements'] != -1]
        df = df[df['number_of_tweet_engagements'] == -1]
        
        ### 4 ###
        tweet_engagement_path = pickle_path + "tweet_id_engagement_new_4.pkl"

        if os.path.exists(tweet_engagement_path ) :
            with open(tweet_engagement_path, 'rb') as f :
                tweet_engagements = pickle.load(f)
                tweet_engagements = defaultdict(lambda : -1, tweet_engagements)

        df['number_of_tweet_engagements'] = df.apply(lambda x : tweet_engagements[x['tweet_id']], axis = 1)
        
        df4 = df[df['number_of_tweet_engagements'] != -1]
        df = df[df['number_of_tweet_engagements'] == -1]
        
        ### 5 ###
        tweet_engagement_path = pickle_path + "tweet_id_engagement_new_5.pkl"

        if os.path.exists(tweet_engagement_path ) :
            with open(tweet_engagement_path, 'rb') as f :
                tweet_engagements = pickle.load(f)
                tweet_engagements = defaultdict(lambda : -1, tweet_engagements)

        df['number_of_tweet_engagements'] = df.apply(lambda x : tweet_engagements[x['tweet_id']], axis = 1)
        
        df5 = df[df['number_of_tweet_engagements'] != -1]
        df = df[df['number_of_tweet_engagements'] == -1]

        ### 6 : from creator ###
        tweet_engagement_path = pickle_path + "creator_avg_all.pkl"

        if os.path.exists(tweet_engagement_path ) :
            with open(tweet_engagement_path, 'rb') as f :
                tweet_engagements = pickle.load(f)
                tweet_engagements = defaultdict(lambda : self.default_values['number_of_tweet_engagements'], tweet_engagements)


        df['number_of_tweet_engagements'] = df.apply(lambda x : tweet_engagements[x['creator_id']], axis = 1)


        df = pd.concat([df1, df2, df3, df4, df5, df])
        del df1, df2, df3
        df = df.reset_index(drop=True)
        
        # creator engagements
        engagements_path_list = ["creator_avg_like.pkl", "creator_avg_retweet.pkl", "creator_avg_reply.pkl", "creator_avg_comment.pkl"]
        feature_list = ['number_of_tweet_like', 'number_of_tweet_retweet', 'number_of_tweet_reply', 'number_of_tweet_comment']

        for i in range(4) :
            tweet_engagement_path = pickle_path + engagements_path_list[i]

            if os.path.exists(tweet_engagement_path ) :
                with open(tweet_engagement_path, 'rb') as f :
                    tweet_engagements = pickle.load(f)
                    tweet_engagements = defaultdict(lambda : self.default_values[feature_list[i]], tweet_engagements)

            df[feature_list[i]] = df.apply(lambda x : tweet_engagements[x['creator_id']], axis = 1)
            del tweet_engagements
        
        return df



    def tweet_features(self, df):
        df['len_text_tokens'] = df['text_tokens'].apply(lambda x: len(x.split('\t')))
        df['len_text_tokens_unique'] = df['text_tokens'].apply(lambda x: len(list(set(x.split('\t')))))
        df['cnt_mention'] = df['text_tokens'].apply(lambda x: (x.split('\t').count('137')))
        df = df.drop('text_tokens', axis=1)
        return df
    
    def set_engager_follows_creator(self, df):
        #### 나중에 피클 나눠지면 더 추가하기 ####
        ### 1 ###
        engager_follows_creator_path = pickle_path + "engager_follows_creator_2.pkl"

        
        if os.path.exists(engager_follows_creator_path ) :
            with open(engager_follows_creator_path, 'rb') as f :
                engager_follows_creator = pickle.load(f)
                engager_follows_creator = defaultdict(lambda : {}, engager_follows_creator)
        df['engager_follows_creator'] = df.apply(lambda x : 1 if len(engager_follows_creator[x['engager_id']]) > 0 and x['creator_id'] in engager_follows_creator[x['engager_id']]  else 0.5, axis = 1)


        df1 = df[df['engager_follows_creator'] != 0.5]
        df = df[df['engager_follows_creator'] == 0.5]

        ### 2 ###
        engager_follows_creator_path = pickle_path + "engager_follows_creator_3.pkl"

        if os.path.exists(engager_follows_creator_path ) :
            with open(engager_follows_creator_path, 'rb') as f :
                engager_follows_creator = pickle.load(f)
                engager_follows_creator = defaultdict(lambda : {}, engager_follows_creator)
        df['engager_follows_creator'] = df.apply(lambda x : 1 if len(engager_follows_creator[x['engager_id']]) > 0 and x['creator_id'] in engager_follows_creator[x['engager_id']]  else 0.5, axis = 1)

        df2 = df[df['engager_follows_creator'] != 0.5]
        df = df[df['engager_follows_creator'] == 0.5]
       
        ### not follow ###
        ### 1 ###
        engager_follows_creator_path = pickle_path + "engager_not_follows_creator_2.pkl"

        if os.path.exists(engager_follows_creator_path ) :
            with open(engager_follows_creator_path, 'rb') as f :
                engager_follows_creator = pickle.load(f)
                engager_follows_creator = defaultdict(lambda : {}, engager_follows_creator)
        df['engager_follows_creator'] = df.apply(lambda x : 0 if len(engager_follows_creator[x['engager_id']]) > 0 and x['creator_id'] in engager_follows_creator[x['engager_id']]  else 0.5, axis = 1)

        df3 = df[df['engager_follows_creator'] != 0.5]
        df = df[df['engager_follows_creator'] == 0.5]
        
        ### 2 ###
        engager_follows_creator_path = pickle_path + "engager_not_follows_creator_3.pkl"

        if os.path.exists(engager_follows_creator_path ) :
            with open(engager_follows_creator_path, 'rb') as f :
                engager_follows_creator = pickle.load(f)
                engager_follows_creator = defaultdict(lambda : {}, engager_follows_creator)
        df['engager_follows_creator'] = df.apply(lambda x : 0 if len(engager_follows_creator[x['engager_id']]) > 0 and x['creator_id'] in engager_follows_creator[x['engager_id']]  else 0.5, axis = 1)

#         df4 = df[df['engager_follows_creator'] != 0.5]
#         df = df[df['engager_follows_creator'] == 0.5]
        
       

        df = pd.concat([df1, df2, df3, df])
        del df1, df2, df3
        df = df.reset_index(drop=True)
        return df
        
    
    







