import numpy as np
from modulo import simtoseis_library as sts
import joblib
import mlflow
import pandas as pd
from pycaret.regression import *

#-----------------------------------------------Training-------------------------------------------------------------------------------

prop_treino = 0.75
n_estimators = 400
max_depth = 40
n_jobs = -1

dict_params = {"n_estimators":n_estimators,"max_depth":max_depth, "n_jobs":n_jobs,"proporcao_treino": prop_treino}

# Carregando dados tratados
sim_clean = np.load("output/sim_clean.npy")

# Convertendo para DataFrame
df = pd.DataFrame(sim_clean, columns=["X", "Y", "Z", "Propriedade"])

# Setando o MLFlow
mlflow.set_tracking_uri('http://127.0.0.1:5000/')
mlflow.set_experiment(experiment_id=584656037219305060)

s = setup(data = df, target = "Propriedade", session_id=123, log_experiment=True, experiment_name="mba-mlops", log_plots=True)

best_model = compare_models()

save_model(best_model, "output/best_model_pycaret")

print('...Auto Ml com Pycaret finalizado!')

