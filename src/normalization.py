import torch
import torch.nn.functional as F
import numpy as np
from train_parameters import *

class Normalization:
    def __init__(self,data):
        self.min = torch.min(data,dim=0)[0]  # min in each column
        self.max = torch.max(data,dim=0)[0]  # max in each column
        self.cols = data.size()[1]    # number of columns
    
    def normalize(self, data):  #normalization
        list_index_cat = []       
        temp = torch.zeros(data.shape,device=data.device)
        for i in range(0, self.cols):
            #scale to [0,1]
            temp[:,i] = torch.div(data[:,i]-self.min[i], self.max[i]-self.min[i])
        return temp

    def unnormalize(self, data):  #denormalization
        temp = torch.zeros(data.shape,device=data.device)
        for i in range(0, self.cols):
            temp[:,i] = torch.mul(data[:,i], self.max[i]-self.min[i]) +self.min[i]
        return temp
