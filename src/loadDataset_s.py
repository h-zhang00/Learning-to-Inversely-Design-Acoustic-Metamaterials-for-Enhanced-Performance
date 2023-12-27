import torch
from torch.utils.data import TensorDataset
import pickle
import numpy as np
import pandas as pd
from train_parameters import *
from src.normalization import Normalization
from src.model_utils import CPU_Unpickler

def exportTensor(name,data,cols, header=True):
    df=pd.DataFrame.from_records(data.detach().numpy())
    if(header):
        df.columns = cols
    print(name)
    df.to_csv(name+".csv", header=header, index=False)

def exportList(name,data):
    arr=np.array(data)
    np.savetxt(name+".csv", [arr], delimiter=',')

def getNormalization_fix(save_normalization, dataPath_range):  # get normalization info

    data = pd.read_csv(dataPath_range)
    # checking for NaNs
    assert not data.isnull().values.any()

    s_range = torch.tensor(data[s_names].values).float()
    s_scaling = Normalization(s_range) # obtaining min, max in each column and column number

    # this should only be activated when retraining with different datasets
    if save_normalization:
        with open('src/normalization/s_scaling_fix.pickle', 'wb') as file_:
            pickle.dump(s_scaling, file_, -1)

    return s_scaling

def getSavedNormalization_fix():
    s_scaling = CPU_Unpickler(open("src/normalization/s_scaling_fix.pickle", "rb", -1)).load()
    return s_scaling

def getDataset(s_scaling, dataPath):

    data = pd.read_csv(dataPath)
    
    print('TrainingData.shape: ',data.shape)
    # checking for NaNs
    assert not data.isnull().values.any()
    #getting column data according to column names, same as 'getNormalization'
    s = torch.tensor(data[s_names].values)
    f = torch.tensor(data[f_names].values)

    s = s_scaling.normalize(s)

    dataset =  TensorDataset(s.float(), f.float())
    l1 = round(len(dataset)*traintest_split)
    l2 = len(dataset) - l1
    print('train/test: ',[l1,l2],'\n\n')
    train_set, test_set = torch.utils.data.random_split(dataset, [l1,l2], generator=torch.Generator().manual_seed(42))
    return train_set, test_set

def getDataset_F1test(s_scaling, dataPath_F1test):
    data = pd.read_csv(dataPath_F1test)

    print('Data: ', data.shape)
    # checking for NaNs
    assert not data.isnull().values.any()

    s = torch.tensor(data[s_names].values)
    f = torch.tensor(data[f_names].values)

    s = s_scaling.normalize(s)

    s_test=s.float()
    f_test=f.float()

    return s_test, f_test

def getDataset_G1predict(dataPath_G1predict):
    data = pd.read_csv(dataPath_G1predict)

    print('Data: ', data.shape)
    f_target = torch.tensor(data[f_target_names].values)
    f_target=f_target.float()

    return f_target