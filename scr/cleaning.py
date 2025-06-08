
import numpy as np
from modulo import simtoseis_library as sts


#-----------------------------------------------Loading-------------------------------------------------------------------------------


# Dados de treino
sim_slice = np.load("data/sim_slice.npy")

# Dados de Inferência
seismic_slice = np.load("data/seismic_slice.npy")

print('...Loading Pronto')


#-----------------------------------------------Cleaning-------------------------------------------------------------------------------

sim_data = sts.simulation_data_cleaning(simulation_data=sim_slice, value_to_clean= -99.0)
sim_data = sts.simulation_nan_treatment(simulation=sim_data, value=0, method='replace')

sim_data, seismic_slice = sts.depth_signal_checking(simulation_data=sim_data, seismic_data=seismic_slice)

np.save("output/sim_clean.npy", sim_data)
np.save("output/seismic_slice_clean.npy", seismic_slice)

print('...Cleaning Pronto')