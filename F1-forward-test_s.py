import pathlib
import torch
from torch.utils.data import DataLoader
from train_parameters import *
from src.loadDataset_s import *
from src.model_utils import *

datafilename = 'data1_1000_S-A_range2'
dataPath_F1test = './data/' + datafilename + '.csv'
modelname = 'F1-range2'

if __name__ == '__main__':
    # load data for preprocessing
    s_scaling = getSavedNormalization_fix()
    s_test, f_test = getDataset_F1test(s_scaling, dataPath_F1test)
    print('\n-------------------------------------')

    f_test_mean = torch.mean(f_test, dim=1, keepdim=True)

    F1_test_history = torch.zeros(len(s_test),1)
    F1 = torch.load('models/'+modelname+'.pt', map_location=device)
    F1.eval()
    print('start')
    f_test_pred = F1(s_test)
    print('finish')
    # calculate mean absorption
    f_test_pred_mean = torch.mean(f_test_pred, dim=1, keepdim=True)
    # calculate mean difference
    f_mean_diff = f_test_pred_mean - f_test_mean

    for i in range(len(f_test)):
        F1_test_loss = lossFn(f_test_pred[i,:],f_test[i,:]).item()
        F1_test_history[i] = F1_test_loss
    exportTensor('data/F1-test/F1_test_' + modelname + '_' + datafilename + '_loss_history', F1_test_history, ['F1-test-loss'])

    ff = torch.cat((f_test_pred, f_test, f_test_pred_mean, f_test_mean, f_mean_diff, F1_test_history), dim=1)

    exportTensor('data/F1-test/F1_test_' + modelname + '_' + datafilename, ff, f_F1_names + f_FEM_names + ['f_test_pred_mean'] + ['f_test_FEM_mean'] + ['f_mean_diff'] + ['f_Fnloss'])
    print('F1 test finished\n-------------------------------------')

