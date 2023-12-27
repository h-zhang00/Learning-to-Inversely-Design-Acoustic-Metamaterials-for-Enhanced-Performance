import pathlib
import torch
from torch.utils.data import DataLoader
from train_parameters import *
from src.loadDataset_s import *
from src.model_utils import *
from visdom import Visdom

if __name__ == '__main__':

    inv_train = True

    modelname = 'F1-range2'
    modelname1 = 'G1-range2'

    dataPath = 'data/data_20000_S-A_range1_smaller0-82.csv'
    dataPath_range = 'data/range2.csv'

    np.random.seed(1234)
    torch.manual_seed(1234)
    
    # directories
    pathlib.Path('models').mkdir(exist_ok=True)
    pathlib.Path('training').mkdir(exist_ok=True)
    pathlib.Path('training/history').mkdir(exist_ok=True)

    # load and preprocess data
    save_normalization = True
    s_scaling= getNormalization_fix(save_normalization, dataPath_range) 
    train_set, test_set = getDataset(s_scaling, dataPath)    
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

    ## load forward model F1
    F1 = torch.load('models/'+modelname+'.pt',map_location=device)
    F1.eval()

    ## inverse model G1
    # create inverse model G1
    G1 = createNN(fpoints,inv_arch,str_para+mat_para).to(device) # 31 to 16
    # optimizer
    inv_optimizer = torch.optim.Adam(G1.parameters(), lr=inv_learning_rate)
    inv_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(inv_optimizer, 'min', patience=20, factor=0.5)
    inv_train_history,inv_test_history = [],[]
    # training G1
    for inv_epoch_iter in range(inv_train_epochs):
        inv_train_loss = 0.        
        for iteration, batch in enumerate(train_data_loader,0):

            f_train = batch[1].to(device)
            # train mode
            G1.train()
            # inverse design
            s_train_pred= G1(f_train)
            # forward pass F1
            f_train_pred_pred = F1(s_train_pred)
            # calculate loss
            invLoss = lossFn(f_train_pred_pred, f_train)
            # optimize
            inv_optimizer.zero_grad()
            invLoss.backward()
            inv_optimizer.step()
            # store (batch) training loss
            inv_train_loss = invLoss.item()
        f_test = f_test.to(device)
        # prediction results using test data
        s_test_pred= G1(f_test)
        f_test_pred_pred = F1(s_test_pred)
        inv_test_loss = lossFn(f_test_pred_pred,f_test).item()
        inv_scheduler.step(inv_test_loss)
        print("| {}:{}/{} | lr: {:.2e} | invEpochTrainLoss: {:.2e} | invEpochTestLoss: {:.2e}".format("inv",inv_epoch_iter, inv_train_epochs, inv_optimizer.param_groups[0]['lr'], inv_train_loss, inv_test_loss))
        inv_train_history.append(inv_train_loss)
        inv_test_history.append(inv_test_loss)

        disp_itw = 20
        if inv_epoch_iter % disp_itw == 0:
            # save models
            torch.save(G1, 'models/'+modelname1+'.pt')
            # export the history of loss
            exportList('training/history/inv_train_history', inv_train_history)
            exportList('training/history/inv_test_history', inv_test_history)
            # visdom Show Results
            tmp_loss_train = inv_train_loss
            tmp_loss_test = inv_test_loss
            LossAll_train.append(tmp_loss_train)
            LossAll_test.append(tmp_loss_test)
            EpochIdx.append(inv_epoch_iter)
            viz.line(
                Y=np.column_stack((np.array(LossAll_train), np.array(LossAll_test))),
                X=np.array(EpochIdx),
                win='Loss',
                opts=dict(title='invG1_Loss', xlabel='invG1_epoch_iter', ylabel='loss', legend=['invG1_train', 'invG1_test']))

    print('invG1 training finished\n-------------------------------------')

    G1.eval()

    ## testing
    with torch.no_grad():

        s_test, f_test = s_test.to(device), f_test.to(device)
        # F1's forward prediction based on structual parameters s_test
        f_test_pred = F1(s_test)
        # G1's inverse prediction
        s_test_pred= invModel_output(G1, f_test)

        # forward prediction based on inversely designed s_test_pred
        f_test_pred_pred = F1(s_test_pred)

        ## export for post-processing
        print('\nExporting:')

        # unnormalize data to original range
        s_test_pred = s_scaling.unnormalize(s_test_pred)

        # push tensors back to cpu
        f_test=f_test.cpu()
        s_test_pred = s_test_pred.cpu()
        f_test_pred = f_test_pred.cpu()
        f_test_pred_pred = f_test_pred_pred.cpu()
        
        # export tensors to .csv
        exportTensor("Training/s_test_pred", s_test_pred, s_names)
        exportTensor("Training/f_test",f_test,f_names)  #ground truth
        exportTensor("Training/f_test_pred",f_test_pred,f_names)  #forward prediction
        exportTensor("Training/f_test_pred_pred",f_test_pred_pred,f_names) #calculated based on inversely design parameters
        print('Finished.')