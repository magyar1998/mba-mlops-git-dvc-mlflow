import numpy as np
from modulo import simtoseis_library as sts
import joblib

#-----------------------------------------------Training-------------------------------------------------------------------------------

prop_treino = 0.75
n_estimators = 200
max_depth = 60
n_jobs = -1

dict_params = {"n_estimators":n_estimators,"max_depth":max_depth, "n_jobs":n_jobs,"proporcao_treino": prop_treino}

# Carregando dados tratados
sim_clean = np.load("output/sim_clean.npy")

# Treinando modelo ML
dados_validacao, y, nrms_teste, r2_teste, mape_teste, modelo, ET, X = sts.ML_model_evaluation(dados_simulacao=sim_clean, modelo="extratrees", dict_params=dict_params)

# Salvando modelo
joblib.dump(ET, "output/model.pkl")

# Salvando dados
np.save("output/X.npy", X)
np.save("output/y.npy", y)

print('...Treino ML Pronto')

