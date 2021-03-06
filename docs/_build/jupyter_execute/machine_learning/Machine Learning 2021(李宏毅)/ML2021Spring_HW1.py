#!/usr/bin/env python
# coding: utf-8

# # Homework 1: COVID-19 Cases Prediction (Regression)

# Author: Heng-Jui Chang
# 
# Slides: https://github.com/ga642381/ML2021-Spring/blob/main/HW01/HW01.pdf  
# Videos (Mandarin): https://cool.ntu.edu.tw/courses/4793/modules/items/172854  
# https://cool.ntu.edu.tw/courses/4793/modules/items/172853  
# Video (English): https://cool.ntu.edu.tw/courses/4793/modules/items/176529
# 
# 
# Objectives:
# * Solve a regression problem with deep neural networks (DNN).
# * Understand basic DNN training tips.
# * Get familiar with PyTorch.

# ## 1. Download Data
# 
# 
# If the Google drive links are dead, you can download data from [kaggle](https://www.kaggle.com/c/ml2021spring-hw1/data), and upload data manually to the workspace.

# In[13]:


tr_path = 'data/covid.train.csv'  # path to training data
tt_path = 'data/covid.test.csv'   # path to testing data

# !gdown --id '19CCyCgJrUxtvgZF53vnctJiOJ23T5mqF' --output covid.train.csv
# !gdown --id '1CE240jLm2npU-tdz81-oVKEF3T2yfT1O' --output covid.test.csv


# ## 2. Import Some Packages

# In[82]:


# PyTorch
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# For data preprocess
import numpy as np
import csv
import os

import pandas as pd
# For plotting

import altair as alt
alt.data_transformers.disable_max_rows()

myseed = 42069  # set a random seed for reproducibility
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(myseed)
torch.manual_seed(myseed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(myseed)


# ## 3. Some Utilities
# 
# You do not need to modify this part.

# In[83]:


def get_device():
    ''' Get device (if GPU is available, use GPU) '''
    return 'cuda' if torch.cuda.is_available() else 'cpu'
    
def plot_learning_curve(loss_record, title=''):
    ''' Plot learning curve of your DNN (train & dev loss) '''
    total_steps = len(loss_record['train'])
    x_1 = list(range(total_steps))
    x_2 = list(x_1[::len(loss_record['train']) // len(loss_record['dev'])])
    data = pd.DataFrame({
        "Training Steps": x_1 + x_2,
        "MSE loss": loss_record['train'] + loss_record['dev'],
        "Type": ['train'] * total_steps + ['dev'] * len(x_2)
    })
    
    return alt.Chart(data, title=f'Learning curve of {title}').mark_line(clip=True).encode(
            x=alt.X("Training Steps:Q"),
            y=alt.Y("MSE loss:Q", scale=alt.Scale(domain=(0., 5.))),
            color=alt.Color("Type:N")
        ).properties(
            width=600,
            height=300
        )

def plot_pred(dv_set, model, device, lim=35., preds=None, targets=None):
    ''' Plot prediction of your DNN '''
    if preds is None or targets is None:
        model.eval()
        preds, targets = [], []
        for x, y in dv_set:
            x, y = x.to(device), y.to(device)
            with torch.no_grad():
                pred = model(x)
                preds.append(pred.detach().cpu())
                targets.append(y.detach().cpu())
        preds = torch.cat(preds, dim=0).numpy()
        targets = torch.cat(targets, dim=0).numpy()
    
    data = pd.DataFrame({
        "ground truth value": targets,
        "predicted value": preds
    })
    
    as_data = pd.DataFrame({
        "ground truth value": [0, 36],
        "predicted value": [0, 36]
    })
    
    return alt.Chart(data, title='Ground Truth v.s. Prediction').mark_point(color="red",clip=True).encode(
            x=alt.X("ground truth value:Q", scale=alt.Scale(domain=(0, lim))),
            y=alt.Y("predicted value:Q", scale=alt.Scale(domain=(0, lim))),
        ).properties(
            width=400,
            height=400,
        ) +\
        alt.Chart(as_data).mark_line(clip=True).encode(
            x=alt.X("ground truth value:Q", scale=alt.Scale(domain=(0, lim))),
            y=alt.Y("predicted value:Q", scale=alt.Scale(domain=(0, lim))),
        ).properties(
            width=400,
            height=400
        )


# ## 4. Preprocess
# 
# We have three kinds of datasets:
# * `train`: for training
# * `dev`: for validation
# * `test`: for testing (w/o target value)

# ### 4.1 Dataset
# 
# The `COVID19Dataset` below does:
# * read `.csv` files
# * extract features
# * split `covid.train.csv` into train/dev sets
# * normalize features
# 
# Finishing `TODO` below might make you pass medium baseline.

# In[108]:


class COVID19Dataset(Dataset):
    ''' Dataset for loading and preprocessing the COVID19 dataset '''
    def __init__(self,
                 path,
                 mode='train',
                 target_only=False):
        self.mode = mode

        # Read data into numpy arrays
        with open(path, 'r') as fp:
            data = list(csv.reader(fp))
            data = np.array(data[1:])[:, 1:].astype(float)
        
        if not target_only:
            feats = list(range(93))
        else:
            # TODO: Using 40 states & 2 tested_positive features (indices = 57 & 75)
            feats = list(range(1, 41)) + [57, 75]

        if mode == 'test':
            # Testing data
            # data: 893 x 93 (40 states + day 1 (18) + day 2 (18) + day 3 (17))
            data = data[:, feats]
            self.data = torch.FloatTensor(data)
        else:
            # Training data (train/dev sets)
            # data: 2700 x 94 (40 states + day 1 (18) + day 2 (18) + day 3 (18))
            target = data[:, -1]
            data = data[:, feats]
            
            # Splitting training data into train & dev sets
            if mode == 'train':
                indices = [i for i in range(len(data)) if i % 10 != 0]
            elif mode == 'dev':
                indices = [i for i in range(len(data)) if i % 10 == 0]
            
            # Convert data into PyTorch tensors
            self.data = torch.FloatTensor(data[indices])
            self.target = torch.FloatTensor(target[indices])

        # Normalize features (you may remove this part to see what will happen)
        self.data[:, 40:] =             (self.data[:, 40:] - self.data[:, 40:].mean(dim=0, keepdim=True))             / self.data[:, 40:].std(dim=0, keepdim=True)

        self.dim = self.data.shape[1]

        print('Finished reading the {} set of COVID19 Dataset ({} samples found, each dim = {})'
              .format(mode, len(self.data), self.dim))

    def __getitem__(self, index):
        # Returns one sample at a time
        if self.mode in ['train', 'dev']:
            # For training
            return self.data[index], self.target[index]
        else:
            # For testing (no target)
            return self.data[index]

    def __len__(self):
        # Returns the size of the dataset
        return len(self.data)


# ### 4.2 DataLoader
# 
# A `DataLoader` loads data from a given `Dataset` into batches.
# 

# In[109]:


def prep_dataloader(path, mode, batch_size, n_jobs=0, target_only=False):
    ''' Generates a dataset, then is put into a dataloader. '''
    dataset = COVID19Dataset(path, mode=mode, target_only=target_only)  # Construct dataset
    dataloader = DataLoader(
        dataset, batch_size,
        shuffle=(mode == 'train'), drop_last=False,
        num_workers=n_jobs, pin_memory=True)                            # Construct dataloader
    return dataloader


# ## 5. Deep Neural Network
# 
# `NeuralNet` is an `nn.Module` designed for regression.
# The DNN consists of 2 fully-connected layers with ReLU activation.
# This module also included a function `cal_loss` for calculating loss.
# 

# In[132]:


class NeuralNet(nn.Module):
    ''' A simple fully-connected deep neural network '''
    def __init__(self, input_dim):
        super(NeuralNet, self).__init__()

        # Define your neural network here
        # TODO: How to modify this model to achieve better performance?
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

        # Mean squared error loss
        self.criterion = nn.MSELoss(reduction='mean')

    def forward(self, x):
        ''' Given input of size (batch_size x input_dim), compute output of the network '''
        return self.net(x).squeeze(1)

    def cal_loss(self, pred, target):
        ''' Calculate loss '''
        # TODO: you may implement L1/L2 regularization here
        regularization_loss = 0
        for param in self.net.parameters():
            regularization_loss += torch.sum(param ** 2)
        return self.criterion(pred, target) + 0.00075 * regularization_loss


# ## 6. Train/Dev/Test

# ### 6.1 Training

# In[133]:


def train(tr_set, dv_set, model, config, device):
    ''' DNN training '''

    n_epochs = config['n_epochs']  # Maximum number of epochs

    # Setup optimizer
    optimizer = getattr(torch.optim, config['optimizer'])(
        model.parameters(), **config['optim_hparas'])

    min_mse = 1000.
    loss_record = {'train': [], 'dev': []}      # for recording training loss
    early_stop_cnt = 0
    epoch = 0
    while epoch < n_epochs:
        model.train()                           # set model to training mode
        for x, y in tr_set:                     # iterate through the dataloader
            optimizer.zero_grad()               # set gradient to zero
            x, y = x.to(device), y.to(device)   # move data to device (cpu/cuda)
            pred = model(x)                     # forward pass (compute output)
            mse_loss = model.cal_loss(pred, y)  # compute loss
            mse_loss.backward()                 # compute gradient (backpropagation)
            optimizer.step()                    # update model with optimizer
            loss_record['train'].append(mse_loss.detach().cpu().item())

        # After each epoch, test your model on the validation (development) set.
        dev_mse = dev(dv_set, model, device)
        if dev_mse < min_mse:
            # Save model if your model improved
            min_mse = dev_mse
            print('Saving model (epoch = {:4d}, loss = {:.4f})'
                .format(epoch + 1, min_mse))
            torch.save(model.state_dict(), config['save_path'])  # Save model to specified path
            early_stop_cnt = 0
        else:
            early_stop_cnt += 1

        epoch += 1
        loss_record['dev'].append(dev_mse)
        if early_stop_cnt > config['early_stop']:
            # Stop training if your model stops improving for "config['early_stop']" epochs.
            break

    print('Finished training after {} epochs'.format(epoch))
    return min_mse, loss_record


# ### 6.2 Validation

# In[134]:


def dev(dv_set, model, device):
    model.eval()                                # set model to evalutation mode
    total_loss = 0
    for x, y in dv_set:                         # iterate through the dataloader
        x, y = x.to(device), y.to(device)       # move data to device (cpu/cuda)
        with torch.no_grad():                   # disable gradient calculation
            pred = model(x)                     # forward pass (compute output)
            mse_loss = model.cal_loss(pred, y)  # compute loss
        total_loss += mse_loss.detach().cpu().item() * len(x)  # accumulate loss
    total_loss = total_loss / len(dv_set.dataset)              # compute averaged loss

    return total_loss


# ### 6.3 Testing

# In[135]:


def test(tt_set, model, device):
    model.eval()                                # set model to evalutation mode
    preds = []
    for x in tt_set:                            # iterate through the dataloader
        x = x.to(device)                        # move data to device (cpu/cuda)
        with torch.no_grad():                   # disable gradient calculation
            pred = model(x)                     # forward pass (compute output)
            preds.append(pred.detach().cpu())   # collect prediction
    preds = torch.cat(preds, dim=0).numpy()     # concatenate all predictions and convert to a numpy array
    return preds


# ## 7. Setup Hyper-parameters
# 
# `config` contains hyper-parameters for training and the path to save your model.

# In[142]:


device = get_device()                 # get the current available device ('cpu' or 'cuda')
os.makedirs('models/hw1', exist_ok=True)  # The trained model will be saved to ./models/
target_only = False                   # TODO: Using 40 states & 2 tested_positive features

# TODO: How to tune these hyper-parameters to improve your model's performance?
config = {
    'n_epochs': 3000,                # maximum number of epochs
    'batch_size': 200,               # mini-batch size for dataloader
    'optimizer': 'Adam',              # optimization algorithm (optimizer in torch.optim)
    'optim_hparas': {                # hyper-parameters for the optimizer (depends on which optimizer you are using)
#         'lr': 0.001,                 # learning rate of SGD
#         'momentum': 0.9              # momentum for SGD
    },
    'early_stop': 500,               # early stopping epochs (the number epochs since your model's last improvement)
    'save_path': 'models/hw1/model.pth'  # your model will be saved here
}


# ## 8. Load data and model

# In[137]:


tr_set = prep_dataloader(tr_path, 'train', config['batch_size'], target_only=target_only)
dv_set = prep_dataloader(tr_path, 'dev', config['batch_size'], target_only=target_only)
tt_set = prep_dataloader(tt_path, 'test', config['batch_size'], target_only=target_only)


# In[138]:


model = NeuralNet(tr_set.dataset.dim).to(device)  # Construct model and move to device


# ## 9. Start Training!

# In[139]:


model_loss, model_loss_record = train(tr_set, dv_set, model, config, device)


# In[140]:


plot_learning_curve(model_loss_record, title='deep model')


# In[141]:


del model
model = NeuralNet(tr_set.dataset.dim).to(device)
ckpt = torch.load(config['save_path'], map_location='cpu')  # Load your best model
model.load_state_dict(ckpt)
plot_pred(dv_set, model, device)  # Show prediction on the validation set


# ## 10. Testing
# The predictions of your model on testing set will be stored at `pred.csv`.

# In[143]:


def save_pred(preds, file):
    ''' Save predictions to specified file '''
    print('Saving results to {}'.format(file))
    with open(file, 'w') as fp:
        writer = csv.writer(fp)
        writer.writerow(['id', 'tested_positive'])
        for i, p in enumerate(preds):
            writer.writerow([i, p])

preds = test(tt_set, model, device)  # predict COVID-19 cases with your model

os.makedirs('preds/hw1', exist_ok=True)
save_pred(preds, 'preds/hw1/pred.csv')         # save prediction file to pred.csv


# ## 11. Hints
# 
# ### 11.1 Simple Baseline
# * Run sample code
# 
# ### 11.2 Medium Baseline
# * Feature selection: 40 states + 2 `tested_positive` (`TODO` in dataset)
# 
# ### 11.3 Strong Baseline
# * Feature selection (what other features are useful?)
# * DNN architecture (layers? dimension? activation function?)
# * Training (mini-batch? optimizer? learning rate?)
# * L2 regularization
# * There are some mistakes in the sample code, can you find them?

# ## 12. Reference
# This code is completely written by Heng-Jui Chang @ NTUEE.  
# Copying or reusing this code is required to specify the original author. 
# 
# E.g.  
# Source: Heng-Jui Chang @ NTUEE (https://github.com/ga642381/ML2021-Spring/blob/main/HW01/HW01.ipynb)
