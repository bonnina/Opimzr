import json
import numpy as np

from state import State
from transport_problem import Transport
from stochastic_problem import Stochastic

# load data
with open('input.json', 'r', encoding='utf-8') as f:
    file = json.load(f)
    production = np.array(file['production'])
    transportation = np.array(file['transportation'])
    additional_cost = np.array(file['additional_cost'])
    capacity = np.array(file['capacity'])
    demand = np.array(file['demand'])
    min_demand = np.array(file['min_demand'])
    max_demand = np.array(file['max_demand'])
    storage = np.array(file['storage'])
    penalty = np.array(file['penalty'])

state = State(production, transportation, additional_cost, demand, capacity, min_demand, max_demand, storage, penalty)

# solve and print results
scenarios = 4

for factory in range(scenarios):
  transport = Transport(factory, state)
  total, matrix = transport.solveLp()
  print(f"\nTransport Problem solution for additional production on the factory №{factory+1}")
  print("Transport Matrix")
  print(matrix)
  print("Production cost: ", transport.getProdCost())
  print("Transport cost: ", transport.getTransportCost())
  print(f"Total cost: {total:.2f}\n")

for factory in range(scenarios):
  stochastic = Stochastic(factory, state)
  total, matrix = stochastic.solveNlp()
  np.set_printoptions(precision=3, suppress=True)
  print(f"\nStochastic Problem solution for additional production on the factory №{factory+1}")
  print("Transport Matrix")
  print(matrix)
  print(f"Production cost: {stochastic.getProdCost():.2f}")
  print(f"Transport cost: {stochastic.getTransportCost():.2f}")
  print(f"Total cost: {total:.2f}")
  print(f"Storage cost: {stochastic.getStorageCosts():.2f}")
  print(f"Deficit penalty cost: {stochastic.getDeficitCost():.2f}\n")