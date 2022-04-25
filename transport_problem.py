import cvxpy as cp
import numpy as np


class Transport:
  def __init__(self, factory, state):
    self.factory = factory
    self.state = state
    self.production = self.state.getProdPrice(self.factory)
    self.transportation = self.state.getTransport()
    capacity = self.state.getProduction(self.factory)
    demand = self.getDemand()
    self.matrix = cp.Variable((state.shape()), integer=True)
    self.prod_func = cp.sum(cp.multiply(cp.sum(self.matrix, axis=0), self.production))
    self.transport_func = cp.sum(cp.multiply(self.transportation, self.matrix))
    self.objective = cp.Minimize(self.prod_func + self.transport_func)
    self.constraints = [self.matrix >= 0, cp.sum(self.matrix, axis=0) == capacity, cp.sum(self.matrix, axis=1) == demand]
    self.problem = cp.Problem(self.objective, self.constraints)
    self.total = self.problem.solve()

  def getDemand(self):
    return np.copy(self.state.demand)

  def getProdCost(self):
    return self.prod_func.value

  def getTransportCost(self):
    return self.transport_func.value

  def solveLp(self):
    return self.total, self.matrix.value
