import numpy as np


class State:
  def __init__(self, production, transportation, additional_cost, demand, capacity,
               min_demand=None, max_demand=None, storage=None, penalty=None):
    self.production = production
    self.additional_cost = additional_cost
    self.transportation = transportation
    self.capacity = capacity
    self.demand = demand
    self.min_demand = min_demand
    self.max_demand = max_demand
    self.penalty = penalty
    self.storage = storage

  def getProdPrice(self, strategy):
    production = np.copy(self.production)
    production[strategy] += self.additional_cost[strategy]
    return production

  def getTransport(self):
    return np.copy(self.transportation)

  def getProduction(self, strategy):
    capacity = np.copy(self.capacity)
    capacity[strategy] += 400
    return capacity

  def shape(self):
    return self.transportation.shape
