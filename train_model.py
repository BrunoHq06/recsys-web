from make_dataset import get_rec_data,generate_matrix,base_split,get_rating,get_anime
from lightfm.evaluation import auc_score
from scipy.sparse import csr_matrix
import numpy as np
def train_model(matrix):
    from lightfm import LightFM
    #Instanciando modelo
    model = LightFM(loss='warp')
    #train e test slipt
    train = base_split(matrix)
    model.fit(matrix, epochs=2, num_threads=2)
    return model


print('Iniciando Treinamento......')

### Inicialização de variáveis
rec_data = get_rec_data()
a_matrix = generate_matrix(rec_data)
test = base_split(a_matrix,'test')
print('Treinando o modelo......')
model = train_model(a_matrix)

#Printando acurácia e precision
print('O modelo foi treinado com sucesso com as seguintes métricas: \n')
print('AUC DE TREINO: {x}'.format(x=np.nanmean(auc_score(model,test))))