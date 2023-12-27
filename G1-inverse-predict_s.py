import pathlib
import torch
from torch.utils.data import DataLoader
from train_parameters import *
from src.loadDataset_s import *
from src.model_utils import *

datafilename = 'data2_500_A_over0-82'
dataPath_G1predict = 'data/' + datafilename + '.csv'
dataPath_range = 'data/range2.csv'
save_normalization = True
s_scaling = getNormalization_fix(save_normalization, dataPath_range)

modelname = 'F1-range2'
modelname1 = 'G1-range2'

if __name__ == '__main__':
    # load data for preprocessing
    s_scaling = getSavedNormalization_fix()
    f_target = getDataset_G1predict(dataPath_G1predict)    #getDataset already includes normalisation
    print('\n-------------------------------------')

    G1 = torch.load('models/'+modelname1+'.pt', map_location=device)
    G1.eval()
    print('start')
    s_G1pred = G1(f_target)
    print('finish')
    s_G1predict = s_scaling.unnormalize(s_G1pred)

    F1 = torch.load('models/'+modelname+'.pt', map_location=device)
    f_F1predict = F1(s_G1pred)

    ff = torch.cat((s_G1predict,f_target,f_F1predict), dim=1)

    exportTensor('data/G1-predict/G1-predict_'+modelname1+'_' + datafilename, ff, s_names + f_target_names + f_F1_names)
    print('G1 predict finished\n-------------------------------------')

