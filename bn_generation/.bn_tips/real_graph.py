
from pgmpy import readwrite
from pgmpy.models import BayesianModel
import pandas as pd
import numpy as np

bifmodel = readwrite.BIF.BIFReader(path="./insurance/insurance.bif")
model = BayesianModel(bifmodel.variable_edges)
model.name = bifmodel.network_name
model.add_nodes_from(bifmodel.variable_names)
a = len(nodes)
w_g = np.zeros((a, a))
for (i, j) in model.edges():
    w_g[list(nodes).index(i), list(nodes).index(j)] = 1
np.save('./insurance/DAG.npy', w_g)

# datadf = pd.read_csv("./insurance/data.csv")
# nodes = datadf.columns
# print(nodes)


# nodes = ['DrivingSkill', 'PropCost', 'HomeBase', 'RiskAversion', 'ILiCost',
#          'DrivQuality', 'SeniorTrain', 'CarValue', 'DrivHist', 'Accident',
#          'ThisCarCost', 'ThisCarDam', 'RuggedAuto', 'OtherCarCost', 'VehicleYear',
#          'Airbag', 'OtherCar', 'MakeModel', 'GoodStudent', 'Mileage',
#          'AntiTheft', 'Cushioning', 'MedCost', 'Theft', 'Age',
#          'SocioEcon', 'Antilock']

# nodes = ['either', 'xray', 'bronc', 'asia', 'dysp', 'smoke', 'tub', 'lung']