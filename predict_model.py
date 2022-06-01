from make_dataset import get_rec_data,generate_matrix,base_split,get_anime
import pandas as pd
import numpy as np


####### Funções ########

def train_model(matrix):
    from lightfm import LightFM
    #Instanciando modelo
    model = LightFM(loss='bpr')
    #train e test slipt
    train = base_split(matrix)
    model.fit(matrix, epochs=100, num_threads=10)
    return model

def generate_predict_info(data,id_array=[],rating_array=[]):
    predict_df = pd.DataFrame({'user_id':9999 ,'anime_id':id_array,'rating':rating_array})
    new_df = data.append(predict_df).reset_index(drop=True)
    return new_df

##### Retreino do modelo #####
def retrain_model(item_list=[],rating_list=[]):
    data = get_rec_data()
    anime = get_anime()
    new_data = generate_predict_info(data,item_list,rating_list)
    new_data.drop_duplicates(inplace=True)
    a_matrix = generate_matrix(new_data)
    a_pivot = generate_matrix(new_data,False)
    n_users, n_items = a_pivot.shape
    print('Treinando o modelo......')
    model = train_model(a_matrix)
    list_final = pd.DataFrame(
    {'anime_id': a_pivot.columns.values, 'y_hat': model.predict(a_matrix.shape[0] - 1, np.arange(n_items))})
    list_final.sort_values('y_hat', ascending=False,inplace=True)
    list_final = list_final.join(anime.set_index('anime_id')['name'], on='anime_id')
    list_final = list_final[~list_final["anime_id"].isin(item_list)]
    return list_final.head(10)

# list_final = pd.DataFrame({'anime_id':a_pivot.columns.values,'y_hat':model.predict(a_matrix.shape[0]-1,np.arange(n_items))})
# print(list_final.sort_values('y_hat',ascending=False))

