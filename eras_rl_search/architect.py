import torch
import numpy as np
import torch.nn as nn
from torch.autograd import Variable


def _concat(xs):
  return torch.cat([x.view(-1) for x in xs])


#class Architect(object):
#
#  def __init__(self, model, args, cluster_rela_dict):
#    self.network_momentum = args.momentum
#    self.network_weight_decay = args.weight_decay
#    self.model = model
#    self.optimizer = torch.optim.Adam([self.model.arch_parameters()],
#        lr=args.arch_learning_rate, betas=(0.5, 0.999), weight_decay=args.arch_weight_decay)
#    self.cluster_rela_dict = cluster_rela_dict
#
#  def step(self, h, t, r, eta, network_optimizer, cluster_rela_dict):
#    self.optimizer.zero_grad()
#    self._backward_step(h, t, r, cluster_rela_dict, updateType="alphas")
#    self.optimizer.step()
#
#
#  def _backward_step(self, h, t, r, cluster_rela_dict, updateType):
#
#    self.model.binarization(tau_state=True)
#    loss, init_time, pos_time, neg_time, loss_time, time_first, time_second, time_third = self.model._loss(h, t, r, updateType, cluster_rela_dict)
#    loss += self.model.args.lamb * self.model.regul
#    loss.backward()
#    self.model.restore()


class Architect(object):

  def __init__(self, model, args):
    self.network_momentum = args.momentum
    self.network_weight_decay = args.weight_decay
    self.model = model
    self.optimizer = torch.optim.Adam([self.model.arch_parameters()],
        lr=args.arch_learning_rate, betas=(0.5, 0.999), weight_decay=args.arch_weight_decay)
    #self.cluster_rela_dict = cluster_rela_dict

  def step(self, h, t, r, eta, network_optimizer, cluster_rela_dict):
    self.optimizer.zero_grad()
    self._backward_step(h, t, r, updateType="alphas")
    self.optimizer.step()

  def _backward_step(self, h, t, r, updateType):
    #print("binarization perfromed")
    self.model.binarization(tau_state=True)
    loss, init_time, pos_time, neg_time, loss_time, time_first, time_second, time_third = self.model._loss(h, t, r, updateType, cluster_rela_dict)
    #loss = self.model._loss(h, t, r, updateType)
    loss += self.model.args.lamb * self.model.regul
    loss.backward()
    self.model.restore()
    
    
    