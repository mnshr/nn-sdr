#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import math

import torch
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

from pytorch_lightning import seed_everything
seed_everything(42, workers=True)
torch.use_deterministic_algorithms(True, warn_only=True)


import dcor

from datetime import datetime

from torch import nn

import argparse

import os 


# In[2]:


parser = argparse.ArgumentParser(description="Running BENN")
parser.add_argument('--model1', default=1, type = int, help = 'model1')
parser.add_argument('--model2', default=1, type = int, help = 'model2')
args = parser.parse_args()
model1 = args.model1
model2 = args.model2


# In[3]:


# Create device agnostic code
device = "cuda" if torch.cuda.is_available() else "cpu"
device


# In[4]:


n=5000

dcor_list=[]
for t in range(1,101):
    x_train=pd.read_csv("./data/model" + str(model1) + "-" + str(model2) + "-" + str(n) + "/x_train_" + str(t) + ".csv")
    x_train=x_train.drop('Unnamed: 0', axis=1)
    y_train=pd.read_csv("./data/model" + str(model1) + "-" + str(model2) + "-" + str(n) + "/y_train_" + str(t) + ".csv")
    y_train=y_train.drop('Unnamed: 0', axis=1)
    x_test=pd.read_csv("./data/model" + str(model1) + "-" + str(model2) + "-" + str(n) + "/x_test_" + str(t) + ".csv")
    x_test=x_test.drop('Unnamed: 0', axis=1)
    y_test=pd.read_csv("./data/model" + str(model1) + "-" + str(model2) + "-" + str(n) + "/y_test_" + str(t) + ".csv")
    y_test=y_test.drop('Unnamed: 0', axis=1)
    z_test=pd.read_csv("./data/model" + str(model1) + "-" + str(model2) + "-" + str(n) + "/z_test_" + str(t) + ".csv")
    z_test=z_test.drop('Unnamed: 0', axis=1)


    n=x_train.shape[0]
    p=x_train.shape[1]
    res_d=1
    m=2
    x_train = torch.tensor(x_train.values).to(torch.float)
    y_train = torch.tensor(y_train.values).to(torch.float)
    x_test = torch.tensor(x_test.values).to(torch.float)
    y_test = torch.tensor(y_test.values).to(torch.float)
    mse_loss = nn.MSELoss()
    
    class nn_dr_reg_model(nn.Module):
        def __init__(self, input_features, output_features, dim_red_features, hidden_units_d, hidden_units_e, dim_red_layers, ens_reg_layers):
            super().__init__()
            model_dim_red=[]
            model_dim_red.append(nn.Linear(in_features=input_features, 
                                        out_features=hidden_units_d))
            model_dim_red.append(nn.ReLU())
            for i in range(1,dim_red_layers):
                model_dim_red.append(nn.Linear(in_features=hidden_units_d, 
                                            out_features=hidden_units_d))
                model_dim_red.append(nn.ReLU())
            model_dim_red.append(nn.Linear(in_features=hidden_units_d, 
                                        out_features=dim_red_features))
            self.dim_red_layer_stack = nn.Sequential(*model_dim_red)

            model_ens_reg=[]
            model_ens_reg.append(nn.Linear(in_features=dim_red_features, out_features=hidden_units_e))
            model_ens_reg.append(nn.ReLU())
            for i in range(1,ens_reg_layers):
                model_ens_reg.append(nn.Linear(in_features=hidden_units_e, out_features=hidden_units_e))
                model_ens_reg.append(nn.ReLU())
            model_ens_reg.append(nn.Linear(in_features=hidden_units_e, out_features=output_features))
            self.ens_reg_layer_stack = nn.Sequential(*model_ens_reg)

        def forward(self, x):
            suff_predictor = self.dim_red_layer_stack(x)
            ens_output = self.ens_reg_layer_stack(suff_predictor)
            return ens_output, suff_predictor


        
    model_nn = nn_dr_reg_model(input_features=p, 
                            output_features=1, 
                            dim_red_features=res_d, 
                            hidden_units_d=300,
                            hidden_units_e=300,
                            dim_red_layers=4, 
                            ens_reg_layers=4
                            ).to(device)
    model_nn
    optimizer = torch.optim.Adam(model_nn.parameters(), 
                                lr=0.001)
    epochs = 200
    x_train, y_train = x_train.to(device), y_train.to(device)
    x_test, y_test = x_test.to(device), y_test.to(device)
    for epoch in range(epochs):
        ### Training
        model_nn.train()
        y_pred_train, y_suff_train = model_nn(x_train) 
        loss = mse_loss(y_pred_train, y_train) 
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        ### Testing
        model_nn.eval()
        y_pred_test, y_suff_test = model_nn(x_test)
        loss_test = mse_loss(y_pred_test, y_test) 
        #dcor_test = dcor.distance_correlation(np.float64(y_suff_test.detach().numpy()),np.float64(z_test))

        if epoch % 25 == 0:
            #print(f"Epoch: {epoch} | Loss: {loss:.5f} | Test Loss: {loss_test:.5f} | Dcor: {dcor_test:.5f}")
            now = datetime.now()
            current_time = now.strftime("%H:%M:%S")
            #print("Current Time =", current_time)
    model_nn.eval()
    with torch.inference_mode():
        y_pred_test, y_suff_test = model_nn(x_test)
    y_suff_test=y_suff_test.numpy()
    dcor_current=dcor.distance_correlation(np.float64(y_suff_test),np.float64(z_test.to_numpy()),method="naive")
    dcor_list.append(dcor_current)
    print(model1, model2, t, dcor_current)
dcor_list_df=pd.DataFrame(dcor_list)
dcor_list_df.to_csv("./results-BENN/result-" + str(model1) + "-" + str(model2) + "-" + str(n) + ".csv")
print(model1, model2, np.mean(dcor_list), np.std(dcor_list))

