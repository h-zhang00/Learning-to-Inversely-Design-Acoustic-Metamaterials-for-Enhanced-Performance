import torch
import torch.nn.functional as F
from train_parameters import *
import pickle, io

class CPU_Unpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == 'torch.storage' and name == '_load_from_bytes':
            return lambda b: torch.load(io.BytesIO(b), map_location='cpu')
        else: return super().find_class(module, name)

def getActivation(activ):
    if(activ == 'relu'):
        sigma = torch.nn.ReLU()
    elif(activ == 'tanh'):
        sigma = torch.nn.Tanh()
    elif(activ == 'sigmoid'):
        sigma = torch.nn.Sigmoid()
    elif(activ == 'leaky'):
        sigma = torch.nn.LeakyReLU()
    elif(activ == 'softplus'):
        sigma = torch.nn.Softplus()
    elif(activ == 'logsigmoid'):
        sigma = torch.nn.LogSigmoid()
    elif(activ == 'elu'):
        sigma = torch.nn.ELU()
    elif(activ == 'gelu'):
        sigma = torch.nn.GELU()
    elif(activ == 'none'):
        sigma = torch.nn.Identity()
    else:
        raise ValueError('Incorrect activation function')
    return sigma

def createNN(inputDim,arch,outputDim,bias=True):
    model = torch.nn.Sequential()
    currDim = inputDim
    layerCount = 1
    activCount = 1
    for i in range(len(arch)):
        if(type(arch[i]) == int):
            model.add_module('layer '+str(layerCount),torch.nn.Linear(currDim,arch[i],bias=bias))
            currDim = arch[i]
            layerCount += 1
        elif(type(arch[i]) == str):
            model.add_module('activ '+str(activCount),getActivation(arch[i]))
            activCount += 1
    model.add_module('layer '+str(layerCount),torch.nn.Linear(currDim,outputDim,bias=bias))
    model.add_module('activ '+str(activCount+1),getActivation('sigmoid'))
    return model

def invModel_output(G1,input):  #input is the targeting acoustic properties
    s_train_pred, m_train_pred = torch.split(G1(input), [str_para, mat_para], dim=1)

    return s_train_pred, m_train_pred
