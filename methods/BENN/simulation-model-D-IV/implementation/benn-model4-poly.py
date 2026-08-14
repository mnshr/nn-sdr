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

import time
from time import process_time


# In[2]:


parser = argparse.ArgumentParser(description="Running BENN")
parser.add_argument('--model1', default=1, type = int, help = 'model1')
parser.add_argument('--model2', default=1, type = int, help = 'model2')
parser.add_argument('--n', default=1000, type = int, help = 'n')
parser.add_argument('--m', default=1, type = int, help = 'm')
parser.add_argument('--l1', default=2, type = int, help = 'l1')
parser.add_argument('--l2', default=2, type = int, help = 'l2')
parser.add_argument('--r1', default=50, type = int, help = 'r1')
parser.add_argument('--r2', default=50, type = int, help = 'r2')
parser.add_argument('--d', default=1, type = int, help = 'd')
parser.add_argument('--t', default=1, type = int, help = 't')
parser.add_argument('--ep', default=50, type = int, help = 'ep')
args = parser.parse_args()
model1 = args.model1
model2 = args.model2
n = args.n
m = args.m
l1 = args.l1
l2 = args.l2
r1 = args.r1
r2 = args.r2
res_d = args.d
t = args.t
ep = args.ep
print(model1, model2, n, m, l1, l2, r1, r2, res_d, t, ep)


# In[3]:


# Create device agnostic code
device = "cuda" if torch.cuda.is_available() else "cpu"
device


# In[4]:


#plt.plot(y_trans_train[:,0], y_trans_train[:,1], 'o', color="C0")


# In[5]:


directory="./results-BENN-unified-std/result-" + str(model1) + "-" + str(model2) + "-" + str(m) + "-" + str(n)
if not os.path.exists(directory):
    os.makedirs(directory)


# In[6]:



#dcor_list=[]
#for t in range(1,101):
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
#res_d=1

t1_start = time.time() 
t2_start = process_time()

x_train = torch.tensor(x_train.values).to(torch.float)
x_test = torch.tensor(x_test.values).to(torch.float)
if m==1:
    y_train = torch.tensor(y_train.values).to(torch.float)
    y_test = torch.tensor(y_test.values).to(torch.float)
else:
    y_trans_train = (y_train - y_train.mean()) / y_train.std()
    y_trans_test = (y_test - y_train.mean()) / y_train.std()
    for i in range(2,m+1):
        y_train_intermediate=y_train**i/math.factorial(i)
        y_test_intermediate=y_test**i/math.factorial(i)
        y_test_intermediate = (y_test_intermediate - y_train_intermediate.mean()) / y_train_intermediate.std()
        y_train_intermediate = (y_train_intermediate - y_train_intermediate.mean()) / y_train_intermediate.std()
        y_trans_train = np.concatenate((y_trans_train,y_train_intermediate), axis=1)
        y_trans_test = np.concatenate((y_trans_test,y_test_intermediate), axis=1)
    y_train = torch.tensor(y_trans_train).to(torch.float)
    y_test = torch.tensor(y_trans_test).to(torch.float)


#print(x_train[:5],y_trans[:5])
mse_loss = nn.MSELoss()
# Build model
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


# Create an instance of BlobModel and send it to the target device
model_nn = nn_dr_reg_model(input_features=p, 
                        output_features=m, 
                        dim_red_features=res_d, 
                        hidden_units_d=r1,
                        hidden_units_e=r2,
                        dim_red_layers=l1, 
                        ens_reg_layers=l2
                        ).to(device)
model_nn
optimizer = torch.optim.Adam(model_nn.parameters(), 
                            lr=0.001)
epochs = ep
x_train, y_train = x_train.to(device), y_train.to(device)
x_test = x_test.to(device)
for epoch in range(epochs):
    ### Training
    model_nn.train()

    # 1. Forward pass
    y_pred_train, y_suff_train = model_nn(x_train) 

    # 2. Calculate loss and accuracy
    loss = mse_loss(y_pred_train, y_train) 

    # 3. Optimizer zero grad
    optimizer.zero_grad()

    # 4. Loss backwards
    loss.backward()

    # 5. Optimizer step
    optimizer.step()

    ### Testing
    model_nn.eval()

    y_pred_test, y_suff_test = model_nn(x_test)
    loss_test = mse_loss(y_pred_test, y_test) 
    #dcor_test = dcor.distance_correlation(np.float64(y_suff_test.detach().numpy()),np.float64(z_test))

    if epoch % 25 == 0:
        #print(f"Epoch: {epoch} | Loss: {loss:.5f} | Test Loss: {loss_test:.5f}")
        now = datetime.now()
        current_time = now.strftime("%H:%M:%S")
        #print("Current Time =", current_time)
model_nn.eval()
with torch.inference_mode():
    y_pred_test, y_suff_test = model_nn(x_test)
y_suff_test=y_suff_test.numpy()

t1_stop = time.time() 
t2_stop = process_time()

y_suff_test_df=pd.DataFrame(y_suff_test)
y_suff_test_df.to_csv("./results-BENN-unified-std/result-" + str(model1) + "-" + str(model2) + "-" + str(m) + "-" + str(n) + "/y_suff_" + str(t) + ".csv")


#dcor_current=dcor.distance_correlation(np.float64(y_suff_test),np.float64(z_test.to_numpy()),method="naive")

#dcor_list.append(dcor_current)
#print(model1, model2, t, dcor_current)

time_use=[t1_stop-t1_start,t2_stop-t2_start]
time_use_df=pd.DataFrame(time_use)
time_use_df.to_csv("./results-BENN-unified-std/result-" + str(model1) + "-" + str(model2) + "-" + str(m) + "-" + str(n) + "/time_" + str(t) + ".csv")

    
#dcor_list_df=pd.DataFrame(dcor_list)
#dcor_list_df.to_csv("./results-BENN-unified-std-new/result-" + str(model1) + "-" + str(model2) + "-" + str(m) + "-" + str(n) + ".csv")
#print(model1, model2, np.mean(dcor_list), np.std(dcor_list))


# In[7]:


#plt.plot(y_suff_test, y_test, 'o')#, color="C0")


# In[8]:


#dcor.distance_correlation(np.float64(y_suff_test),np.float64(z_test.to_numpy()[:,1]),method="naive")


# In[9]:


#y_suff_test_df=pd.DataFrame(y_suff_test)
#y_pred_test_df=pd.DataFrame(y_pred_test)


# In[10]:


#dcor.distance_correlation(y_suff_test,z_test)


# In[11]:


#dcor.distance_correlation(np.float64(y_suff_test),np.float64(z_test.to_numpy()))


# In[ ]:





# In[12]:


#dcor.distance_correlation(np.float64(y_suff_test.detach().numpy()),np.float64(z_test.to_numpy()),method="avl")


# In[13]:


#dcor.distance_correlation(np.float64(y_suff_test.detach().numpy()),np.float64(z_test.to_numpy()),method="naive")


# In[14]:


#dcor.distance_correlation(np.float64(z_test.to_numpy()[:,1]),np.float64(z_test.to_numpy()),method="naive")


# In[ ]:




