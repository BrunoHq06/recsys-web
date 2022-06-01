import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix

def get_rec_data():

    import os
    cwd = os.getcwd()

    #Vou usar um dataset oriundo do site my anime list para criar o sistema de recomendação de animes.
    df_anime = pd.read_csv("item.csv")
    df_ratings = pd.read_csv("rating.csv")

    #alguns caras vieram com rating -1, retirei eles da base
    #df_ratings = df_ratings[df_ratings['rating'] >0]
    df_ratings['rating'] = np.where(df_ratings['rating'] < 0, 0, df_ratings['rating'])
    df_ratings = df_ratings.reset_index(drop=True)
    df_ratings = df_ratings.drop_duplicates()

    #tirar users com numero baixo
    mean = df_anime['members'].mean()
    df_anime_mew = df_anime[['name', 'members', 'anime_id']]
    df_anime_mew = df_anime_mew[df_anime_mew['members'] >= mean]
    anime_filter = df_anime_mew['anime_id'].drop_duplicates().tolist()

    #fazer uma cópia da base por precaução
    df_ratings = df_ratings.drop_duplicates(subset=['anime_id','user_id'],keep='last')
    df_bkp = df_ratings

    #vamos diminuir um pouco mais a base e colocar animes que tiveram pelo menos 20 avaliações
    #e usuários ativos, que serão usuários que já viram pelo menos 3 animes

    # anime_votes = df_bkp.groupby('anime_id')['rating'].agg('count')
    # active_user = df_bkp.groupby('user_id')['rating'].agg('count')
    # user_index = active_user[active_user > 3]
    # anime_index = anime_votes[anime_votes > 3]
    # df_bkp = df_bkp[df_bkp['user_id'].isin(user_index)]
    # df_bkp = df_bkp[df_bkp['anime_id'].isin(anime_filter)]

    df_bkp = df_bkp.head(df_bkp[df_bkp['user_id']==500].tail(1).index[0])

    # balanceamento de classes
    # x = df_bkp.iloc[:, 0:2]
    # y = df_bkp['rating']

    # from imblearn.under_sampling import RandomUnderSampler
    # undersample = RandomUnderSampler(sampling_strategy='majority')
    # X_over, y_over = undersample.fit_resample(x, y)

    # test_under = pd.DataFrame(columns=df_bkp.columns)
    # test_under['user_id'] = X_over['user_id']
    # test_under['anime_id'] = X_over['anime_id']
    # test_under['rating'] = y_over

    # df_bkp = test_under

    return df_bkp

def get_anime():
    df_anime = pd.read_csv("item.csv")
    return df_anime

def get_rating():
    import os
    cwd = os.getcwd()
    df_ratings = pd.read_csv("rating.csv")
    return df_ratings

def generate_matrix(data,csr=True):
    matrix = data.pivot(index='user_id', columns='anime_id', values='rating')
    matrix = matrix.fillna(0)
    csr_final = csr_matrix(matrix.values)
    return csr_final if csr==True else matrix

def base_split(matrix,type='train'):
    from lightfm.cross_validation import random_train_test_split
    cross_val = random_train_test_split(matrix, .2)
    return cross_val[0] if type =='train' else cross_val[1]


