#!/usr/bin/env python
# coding: utf-8

# In[29]:


import numpy as np
import pandas as pd
import math

import torch
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

from pytorch_lightning import seed_everything
seed_everything(42, workers=True)
torch.use_deterministic_algorithms(True, warn_only=True)


from scipy import stats

from datetime import datetime

from torch import nn

import argparse

import os 


# In[30]:


parser = argparse.ArgumentParser(description="Running BENN")
parser.add_argument('--model1', default=5, type = int, help = 'model1')
parser.add_argument('--model2', default=3, type = int, help = 'model2')
parser.add_argument('--n', default=10000, type = int, help = 'n')
parser.add_argument('--m', default=2, type = int, help = 'm')
parser.add_argument('--l2', default=2, type = int, help = 'l2')
parser.add_argument('--r2', default=100, type = int, help = 'r2')
parser.add_argument('--d', default=2, type = int, help = 'd')
parser.add_argument('--t', default=1, type = int, help = 't')
args = parser.parse_args()
model1 = args.model1
model2 = args.model2
n = args.n
m = args.m
l2 = args.l2
r2 = args.r2
res_d = args.d
t = args.t
print(model1, model2, n, m, l2, r2, res_d, t)


# In[31]:


# Create device agnostic code
device = "cuda" if torch.cuda.is_available() else "cpu"
device


# In[32]:


directory="./results-BENN-linear/result-" + str(model1) + "-" + str(model2) + "-" + str(m) + "-" + str(n)
if not os.path.exists(directory):
    os.makedirs(directory)


# In[33]:


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
        mean_inter = y_train_intermediate.mean()
        sd_inter = y_train_intermediate.std()
        y_train_intermediate = (y_train_intermediate - mean_inter) / sd_inter
        y_trans_train = np.concatenate((y_trans_train,y_train_intermediate), axis=1)
        y_test_intermediate=y_test**i/math.factorial(i)
        y_test_intermediate = (y_test_intermediate - mean_inter) / sd_inter
        y_trans_test = np.concatenate((y_trans_test,y_test_intermediate), axis=1)
    y_train = torch.tensor(y_trans_train).to(torch.float)
    y_test = torch.tensor(y_trans_test).to(torch.float)


# In[34]:






#print(x_train[:5],y_trans[:5])
mse_loss = nn.MSELoss()
# Build model
class nn_dr_reg_model(nn.Module):
    def __init__(self, input_features, output_features, dim_red_features, hidden_units_e, ens_reg_layers):
        super().__init__()
        model_dim_red=[]
        model_dim_red.append(nn.Linear(in_features=input_features, 
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
                        hidden_units_e=r2,
                        ens_reg_layers=l2
                        ).to(device)
model_nn
optimizer = torch.optim.Adam(model_nn.parameters(), 
                            lr=0.001)
epochs = 300
x_train, y_train = x_train.to(device), y_train.to(device)
x_test, y_test = x_test.to(device), y_test.to(device)
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
    #corr_test = max(abs(stats.pearsonr(np.float64(y_suff_test.detach().numpy()[:,1]),np.float64(z_test.to_numpy()[:,1])).statistic),
    #                abs(stats.pearsonr(np.float64(y_suff_test.detach().numpy()[:,0]),np.float64(z_test.to_numpy()[:,1])).statistic))

    if epoch % 25 == 0:
        print(f"Epoch: {epoch} | Loss: {loss:.5f} | Test loss: {loss_test:.5f}")
        now = datetime.now()
        current_time = now.strftime("%H:%M:%S")
        print("Current Time =", current_time)
model_nn.eval()
with torch.inference_mode():
    y_pred_test, y_suff_test = model_nn(x_test)
y_suff_test=y_suff_test.numpy()
#if res_d==2:
#    corr_current=[abs(stats.pearsonr(np.float64(y_suff_test[:,0]),np.float64(z_test.to_numpy()[:,0])).statistic),
#                  abs(stats.pearsonr(np.float64(y_suff_test[:,0]),np.float64(z_test.to_numpy()[:,1])).statistic),
#                  abs(stats.pearsonr(np.float64(y_suff_test[:,1]),np.float64(z_test.to_numpy()[:,0])).statistic),
#                  abs(stats.pearsonr(np.float64(y_suff_test[:,1]),np.float64(z_test.to_numpy()[:,1])).statistic)]
#corr_list.append(corr_current)
#print(model1, model2, t, corr_current)
#corr_list_df=pd.DataFrame(corr_list)
#corr_list_df.to_csv("./results-BENN-linear/result-" + str(model1) + "-" + str(model2) + "-" + str(m) + "-" + str(n) + ".csv")
#print(model1, model2, np.mean(corr_list), np.std(corr_list))


# In[35]:


y_suff_test_df=pd.DataFrame(y_suff_test)
y_suff_test_df.to_csv("./results-BENN-linear/result-" + str(model1) + "-" + str(model2) + "-" + str(m) + "-" + str(n) + "/y_suff_" + str(t) + ".csv")


# In[36]:


#from sklearn.linear_model import LinearRegression
#model=LinearRegression()
#x=y_suff_test
#y=z_test.to_numpy()[:,0]
#model.fit(x, y)
#r_sq1 = model.score(x, y)
#print(f"coefficient of determination: {r_sq1}")
#model=LinearRegression()
#x=y_suff_test
#y=z_test.to_numpy()[:,1]
#model.fit(x, y)
#r_sq2 = model.score(x, y)
#print(f"coefficient of determination: {r_sq2}")


# In[37]:


#r_sq=[r_sq1,r_sq2]
#r_sq_df=pd.DataFrame(r_sq)
#r_sq_df.to_csv("./results-BENN-linear/result-" + str(model1) + "-" + str(model2) + "-" + str(m) + "-" + str(n) + "/r_sq_" + str(t) + ".csv")


# In[10]:


#dcor.distance_correlation(np.float64(y_suff_test.detach().numpy()),np.float64(z_test.to_numpy()),method="naive")


# In[11]:


#plt.plot(y_suff_test[:,0],z_test.to_numpy()[:,1],'o')


# In[12]:


#plt.plot(z_test.to_numpy()[:,0],z_test.to_numpy()[:,1],'o')


# In[ ]:




