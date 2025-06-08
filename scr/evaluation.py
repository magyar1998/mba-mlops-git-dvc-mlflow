import numpy as np
from modulo import simtoseis_library as sts


#-----------------------------------------------Avaliação-------------------------------------------------------------------------------

# Carregando dados de Treino

sim_clean = np.load("output/sim_clean.npy")

# Carregando dados de Inferência

seis_estimated = np.load("output/seis_estimated.npy")

#Carrgando Dados de Referência para a modelagem do (Software Comercial)

seismic_slice_GT = np.load("data/seismic_slice_GT.npy")

print("...Loanding Pronto")

seismic_slice_residuos_final = sts.residuos_calculation(seismic_slice_GT, seis_estimated)

# Plotando dos Histogramas de distribuições

sts.plot_simulation_distribution(sim_clean, bins=35, title="Distribuição da Propriedade da Simulação para os dados de Treino")


sts.plot_simulation_distribution(seis_estimated, bins=35, title="Distribuição da Propriedade da Simulação para os dados Estimados")


sts.plot_simulation_distribution(seismic_slice_residuos_final, bins=35, title="Distribuição da Propriedade da Simulação para os Resíduos")


# Plot das Imagens


sts.plot_seismic_slice(sim_clean, title="Slice a ~5000m dos dados de treino")


sts.plot_seismic_slice(seismic_slice_GT, title="Slice a ~5000m do Resultado-Referência(software comercial)")


sts.plot_seismic_slice(seis_estimated, title="Slice a ~5000m da Inferência ML")


sts.plot_seismic_slice(seismic_slice_residuos_final, title = "Slice a ~5000m - Residuo da Inferência")


np.save("output/residuos.npy", seismic_slice_residuos_final)

print("...Evalation e Ploting Pronto")
