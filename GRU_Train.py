import torch
from torch.utils.data import Dataset
import numpy as np
import pandas as pd

##function to split into sequences
def generate_sequences(df: pd.DataFrame, tw: int, pw: int, target_columns, drop_targets=False):
  '''
  df: Pandas DataFrame of the univariate time-series
  tw: Training Window - Integer defining how many steps to look back
  pw: Prediction Window - Integer defining how many steps forward to predict

  returns: dictionary of sequences and targets for all sequences
  '''
  data = dict() # Store results into a dictionary
  L = len(df)
  for i in range(L-tw-pw):
    # Option to drop target from dataframe
    if drop_targets:
      df.drop(target_columns, axis=1, inplace=True)

    # Get current sequence  
    sequence = df[i:i+tw].values
    # Get values right after the current sequence
    target = df[i+tw:i+tw+pw][target_columns].values
    data[i] = {'sequence': sequence, 'target': target}
  return data

##dataloader class
class SequenceDataset(Dataset):

  def __init__(self, df):
    self.data = df

  def __getitem__(self, idx):
    sample = self.data[idx]
    return torch.from_numpy(sample['sequence']), torch.from_numpy(sample['target'])
  
  def __len__(self):
    return len(self.data)
  
  
def var_loss(pred, target):
  vpred = torch.var(pred, dim = 1)
  vtarg = torch.var(target, dim = 1)
  res = torch.mean(torch.abs(vpred-vtarg))
  return res

## train function
def train(train_loader, learn_rate, pred_window, batch_size = 32, hidden_dim=256, EPOCHS=10, w_var = 5):
    device = torch.device("cuda:0")
    # Setting common hyperparameters
    input_dim = next(iter(train_loader))[0].shape[2]
    output_dim = pred_window
    n_layers = 3
    # Instantiating the models
    model = GRUNet(input_dim, hidden_dim, output_dim, n_layers)

    model.to(device)
    
    # Defining loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learn_rate)
    
    model.train()
    print("Starting Training of {} model".format("GRU"))
    epoch_times = []
    
    # Start training loop
    for epoch in range(1,EPOCHS+1):
        h = model.init_hidden(batch_size)
        #print("H:", h.shape)
        avg_loss = 0.
        counter = 0
        for x, target in train_loader:
            #print(".")
            counter += 1
            h = h.data
            model.zero_grad()
            
            out, h = model(x.to(device).float(), h)
            loss = criterion(out, target.to(device).float()) + w_var * var_loss(out, target.to(device).float())
            loss.backward()
            optimizer.step()
            avg_loss += loss.item()
            if counter%200 == 0:
                print("Epoch {}......Step: {}/{}....... Average Loss for Epoch: {}".format(epoch, counter, len(train_loader), avg_loss/counter), flush = True)
        #current_time = time.clock()
        print("Epoch {}/{} Done, Total Loss: {}".format(epoch, EPOCHS, avg_loss/len(train_loader)),flush = True)
        #print("Total Time Elapsed: {} seconds".format(str(current_time-start_time)))
        #epoch_times.append(current_time-start_time)
    #print("Total Training Time: {} seconds".format(str(sum(epoch_times))))
    return model
  
def prep_data(df: pd.DataFrame, lookback: int, pred_window: int, batch_size: int):
  data_raw = generate_sequences(df, lookback, pred_window, "Wind")
  split = 0.8 # Train/Test Split ratio
  dataset = SequenceDataset(data_raw)
  train_len = int(len(dataset)*split)
  lens = [train_len, len(dataset)-train_len]
  train_ds, test_ds = torch.utils.data.random_split(dataset, lens)
  trainloader = torch.utils.data.DataLoader(train_ds, batch_size=batch_size, shuffle = True, drop_last = True)
  testloader = torch.utils.data.DataLoader(test_ds, batch_size=batch_size, shuffle = False, drop_last = True)
  return trainloader, testloader

###################################################
covs = r.df
wind = r.wind_data
BATCHSIZE = 32
results = dict()
var_weight = [10,8,6,1,0.1,0]

##iterate through all decompositions, train model, save, and make prediction
for ceedman in range(5):
  print("Training model on component",ceedman)
  dat = covs.copy()
  dat['Wind'] = wind[:,ceedman]
  trainload, testload = prep_data(dat, 144, 12, BATCHSIZE)
  trained_model = train(train_loader=trainload, learn_rate = 0.0001, pred_window = 12, batch_size = BATCHSIZE, hidden_dim=256, EPOCHS=25, w_var = var_weight[ceedman])
  mod_name = "GRU_Mod_C" + str(ceedman)
  torch.save(trained_model,mod_name)
  
  ##Prediction
  test_dat = next(iter(testload))
  h = trained_model.init_hidden(BATCHSIZE)
  pred = trained_model(test_dat[0].to("cuda:0").float(), h)
  pred_dat = pred[0].cpu().detach().numpy()
  ground_truth = test_dat[1].numpy()
  temp = {'preds': pred_dat, 'truth': ground_truth}
  results[ceedman] = temp


###Prediction




#####training##############################
# i = 0
# for x, target in trainloader:
#   print(x.shape)
#   print(target.shape)
#   i = i+1
#   if(i > 14):
#     break
#   
#data = next(iter(testload))
#data3 = data[1]

# x = data[0]
# target = data[1]
# input_dim = next(iter(trainloader))[0].shape[2]
# output_dim = 12
# n_layers = 3
# hidden_dim=256
# # Instantiating the models
# model = GRUNet(input_dim, hidden_dim, output_dim, n_layers).to("cuda:0")
# h = model.init_hidden(BATCH_SIZE)
# h = h.data
# model.zero_grad()
# 
# out, h = model(x.to("cuda:0").float(), h)
# criterion = nn.MSELoss()
# 
# loss = criterion(out, target.to(device).float())





# def evaluate(model, test_x, test_y, label_scalers):
#     model.eval()
#     outputs = []
#     targets = []
#     for i in test_x.keys():
#         inp = torch.from_numpy(np.array(test_x[i]))
#         labs = torch.from_numpy(np.array(test_y[i]))
#         h = model.init_hidden(inp.shape[0])
#         out, h = model(inp.to(device).float(), h)
#         outputs.append(label_scalers[i].inverse_transform(out.cpu().detach().numpy()).reshape(-1))
#         targets.append(label_scalers[i].inverse_transform(labs.numpy()).reshape(-1))
#     sMAPE = 0
#     for i in range(len(outputs)):
#         sMAPE += np.mean(abs(outputs[i]-targets[i])/(targets[i]+outputs[i])/2)/len(outputs)
#     print("sMAPE: {}%".format(sMAPE*100))
#     return outputs, targets, sMAPE
