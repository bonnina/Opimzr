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
scenarios = 4

# solve and save results to a file
with open('output.txt', 'w') as f:
    print('Results:\n', file=f)

for factory in range(scenarios):
  transport = Transport(factory, state)
  total, matrix = transport.solve()
  with open('output.txt', 'a') as f:
    print('Transport Problem solution for additional production on the factory №', {factory + 1}, file=f)
    print('Transport Matrix:\n', matrix, file=f)
    print(f"Production cost: {transport.getProdCost():.2f}", file=f)
    print(f"Transport cost: {transport.getTransportCost():.2f}", file=f)
    print(f"Total cost: {total:.2f}\n", file=f)

for factory in range(scenarios):
    stochastic = Stochastic(factory, state)
    total, matrix = stochastic.solve()
    np.set_printoptions(precision=3, suppress=True)
    with open('output.txt', 'a') as f:
        print('Stochastic Problem solution for additional production on the factory №', {factory + 1}, file=f)
        print('Transport Matrix:\n', matrix, file=f)
        print(f"Production cost: {stochastic.getProdCost():.2f}", file=f)
        print(f"Transport cost: {stochastic.getTransportCost():.2f}", file=f)
        print(f"Total cost: {total:.2f}", file=f)
        print(f"Storage cost: {stochastic.getStorageCosts():.2f}", file=f)
        print(f"Deficit penalty cost: {stochastic.getDeficitCost():.2f}\n", file=f)
