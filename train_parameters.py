import torch

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# training data path
#dataPath = './data/traindata-w.csv'   #'./data/training.csv'

str_para=10 # the number of structural parameters, = 10
mat_para=0  # the number of material parameters, which is 0 in this project
fpoints=109 # the number of frequency points, = 109

# define the train/test split ratio
traintest_split = 0.99

# define batch size and numWorkers
batchSize = 200  
numWorkers = 0

## define the network architecture
# F1 forward training
F1_train_epochs = 200000 #3000
F1_arch = [128,'leaky',64,'leaky',64,'leaky',32,'leaky',32,'leaky']
F1_learning_rate = 1e-3

# G1 inverse training
inv_train_epochs = 200000  #3000
inv_arch = [1024,'leaky',512,'leaky',256,'leaky',128,'leaky',64,'leaky',32,'leaky']
inv_learning_rate = 5e-4

# define the loss function
lossFn = torch.nn.MSELoss()

# define column names according to the data file
s_names = []
f_names = []
f_target_names = []
f_F1_names = []
f_FEM_names = []

for i in range(str_para):
    tmp='s'+str(i+1)
    s_names.append(tmp)

for i in range(fpoints):
    tmp='f'+str(i+1)
    f_names.append(tmp)

for i in range(fpoints):
    tmp='f_F1_'+str(i+1)
    f_F1_names.append(tmp)

for i in range(fpoints):
    tmp='f_target'+str(i+1)
    f_target_names.append(tmp)

for i in range(fpoints):
    tmp='f_FEM'+str(i+1)
    f_FEM_names.append(tmp)

all_names = s_names + f_names

