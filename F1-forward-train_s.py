import pathlib
import torch
from torch.utils.data import DataLoader
from train_parameters import *
from src.loadDataset_s import *
from src.model_utils import *
from visdom import Visdom

if __name__ == '__main__':

    modelname = 'F1-range2'
    dataPath = 'data/data_20000_S-A_range1_smaller0-82.csv'
    dataPath_range = 'data/range2.csv'

    np.random.seed(1234)
    torch.manual_seed(1234)
    
    # directories
    pathlib.Path('models').mkdir(exist_ok=True)
    pathlib.Path('training').mkdir(exist_ok=True)

    # load and preprocess data
    save_normalization = True
    s_scaling= getNormalization_fix(save_normalization, dataPath_range) #obtaining the range of each structural parameter
    train_set, test_set = getDataset(s_scaling, dataPath)    #get the normalized dataset
    train_data_loader = DataLoader(dataset=train_set, num_workers=numWorkers, batch_size=batchSize)
    test_data_loader = DataLoader(dataset=test_set, num_workers=numWorkers, batch_size=len(test_set))

    s_test, f_test = next(iter(test_data_loader))
    print('\n-------------------------------------')

    # visdom
    viz = Visdom(port=8098)
    LossAll_train = []
    LossAll_test = []
    EpochIdx = []
    # python -m visdom.server -p 8098
    # internet explorer localhost:8098

    ## first forward model (F1)
    # create first forward model F1
    F1 = createNN(str_para+mat_para,F1_arch,fpoints).to(device)   #10 to 109
    # optimizer
    F1_optimizer = torch.optim.Adam(F1.parameters(), lr=F1_learning_rate)
    F1_train_history, F1_test_history = [],[]
    # training
    for F1_epoch_iter in range(F1_train_epochs):
        F1_train_loss = 0.
        for iteration, batch in enumerate(train_data_loader,0):
            # get batch
            s_train, f_train = batch[0].to(device), batch[1].to(device)
            F1.train()
            # F1 forward pass
            f_train_pred = F1(s_train)
            # calculate loss
            fwdLoss = lossFn(f_train_pred,f_train)
            # optimize
            F1_optimizer.zero_grad()
            fwdLoss.backward()
            F1_optimizer.step()
            # store training loss
            F1_train_loss = fwdLoss.item()
        s_test, f_test = s_test.to(device), f_test.to(device)
        f_test_pred = F1(s_test)

        F1_test_loss = lossFn(f_test_pred,f_test).item()
        print("| {}:{}/{} | F1_EpochTrainLoss: {:.2e} | F1_EpochTestLoss: {:.2e}".format("F1",F1_epoch_iter,F1_train_epochs,F1_train_loss,F1_test_loss))
        F1_train_history.append(F1_train_loss)
        F1_test_history.append(F1_test_loss)
        
        disp_itw = 20
        if F1_epoch_iter % disp_itw == 0:
            # save model as 'F1-range2'
            torch.save(F1, 'models/'+modelname+'.pt')
            # export the history of loss
            exportList('training/'+modelname+'_train_history', F1_train_history)
            exportList('training/'+modelname+'_test_history', F1_test_history)
            # visdom Show Results
            tmp_loss_train = F1_train_loss
            tmp_loss_test = F1_test_loss
            LossAll_train.append(tmp_loss_train)
            LossAll_test.append(tmp_loss_test)
            EpochIdx.append(F1_epoch_iter)
            viz.line(
                Y=np.column_stack((np.array(LossAll_train), np.array(LossAll_test))),
                X=np.array(EpochIdx),
                win='Loss',
                opts=dict(title='F1_Loss', xlabel='F1_epoch_iter', ylabel='loss', legend=['F1_train', 'F1_test']))

    print('F1 training finished\n-------------------------------------')

    F1.eval()
