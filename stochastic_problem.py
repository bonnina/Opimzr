import cvxpy as cp
import numpy as np


class Stochastic:
  def __init__(self, factory, state):
    self.factory = factory
    self.state = state
    transportation = self.state.getTransport()
    production = self.state.getProdPrice(self.factory)
    capacity = self.state.getProduction(self.factory)
    min_demand = self.getMinDemand()
    max_demand = self.getMaxDemand()
    mean_demand = (min_demand + max_demand) / 2
    storage = self.getStorageCost()
    penalty = self.getPenalty()
    mean_storage = ((storage + penalty) / (max_demand - min_demand))
    self.matrix = cp.Variable((self.state.shape()), integer=True)
    self.transport_func = cp.sum(cp.multiply(transportation, self.matrix))
    self.prod_func = cp.sum(cp.multiply(cp.sum(self.matrix, axis=0), production))
    self.storage_cost_func = cp.sum(cp.multiply(mean_storage, cp.square(cp.sum(self.matrix, axis=1) - min_demand)))
    self.deficit_penalty_func = cp.sum(cp.multiply(penalty, (mean_demand - cp.sum(self.matrix, axis=1))))
    self.objective = cp.Minimize(self.prod_func + self.transport_func + self.deficit_penalty_func + self.storage_cost_func)
    self.constraints = [self.matrix >= 0, cp.sum(self.matrix, axis=0) == capacity, cp.sum(self.matrix, axis=1) >= min_demand,
                        cp.sum(self.matrix, axis=1) <= max_demand]
    self.problem = cp.Problem(self.objective, self.constraints)
    self.total = self.problem.solve(solver='ECOS_BB')

  def getMinDemand(self):
    return np.copy(self.state.min_demand)

  def getMaxDemand(self):
    return np.copy(self.state.max_demand)

  def getStorageCost(self):
    return np.copy(self.state.storage)

  def getPenalty(self):
    return np.copy(self.state.penalty)

  def getProdCost(self):
    return self.prod_func.value

  def getTransportCost(self):
    return self.transport_func.value

  def getDeficitCost(self):
    return self.deficit_penalty_func.value

  def getStorageCosts(self):
    return self.storage_cost_func.value

  def solveNlp(self):
    return self.total, self.matrix.value
