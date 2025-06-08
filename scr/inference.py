import numpy as np
from modulo import simtoseis_library as sts
import joblib

#-----------------------------------------------Inferência-------------------------------------------------------------------------------

# Carregando dados para Inferência

seismic_slice_clean = np.load("output/seismic_slice_clean.npy")
X = np.load("output/X.npy")
y = np.load("output/y.npy")

# Carrgando modelo
ET = joblib.load("output/model.pkl")

# Realizando a Inferência
seis_prop_vector, seis_estimated = sts.transfer_to_seismic_scale(dados_sismicos=seismic_slice_clean, nome_arquivo_segy=None, modelo=ET, X=X, y=y)

np.save("output/seis_estimated.npy", seis_estimated)

print("...Inferência Pronto")