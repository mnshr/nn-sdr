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
parser.add_argument('--model1', default=4, type = int, help = 'model1')
parser.add_argument('--model2', default=1, type = int, help = 'model2')
parser.add_argument('--n', default=5000, type = int, help = 'n')
parser.add_argument('--m', default=1000, type = int, help = 'm')
parser.add_argument('--l1', default=2, type = int, help = 'l1')
parser.add_argument('--l2', default=1, type = int, help = 'l2')
parser.add_argument('--r1', default=50, type = int, help = 'r1')
parser.add_argument('--r2', default=2000, type = int, help = 'r2')
parser.add_argument('--d', default=2, type = int, help = 'd')
parser.add_argument('--t', default=3, type = int, help = 't')
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
print(model1, model2, n, m, l1, l2, r1, r2, res_d, t)


# In[3]:


directory="./results-BENN-unified-std/result-" + str(model1) + "-" + str(model2) + "-" + str(m) + "-" + str(n)
if not os.path.exists(directory):
    os.makedirs(directory)


# In[ ]:





# In[4]:


# Create device agnostic code
device = "cuda" if torch.cuda.is_available() else "cpu"
device


# In[5]:


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
y_train_original=y_train
y_test_original=y_test


# In[ ]:





# In[6]:


x_train=x_train.to_numpy()
y_train=y_train.to_numpy()
x_test=x_test.to_numpy()
y_test=y_test.to_numpy()


# In[7]:


t1_start = time.time() 
t2_start = process_time()


# In[8]:


def kernel_func(bw, y1, y2):
    return math.exp(-((y1-y2)/bw)**2/2)


# In[9]:


y_list=np.random.uniform(low=y_train.mean()-2*y_train.std(),
                         high=y_train.mean()+2*y_train.std(),
                         size=m)
bw=y_train.std()


# In[10]:


y_trans_train=[]
for i in range(y_train.shape[0]):
    y_trans_current=[]
    for j in range(m):
        y_trans_current.append(kernel_func(bw,y_list[j],y_train[i][0]))
    y_trans_train.append(y_trans_current)
y_trans_train=np.array(y_trans_train)
y_trans_test=[]
for i in range(y_test.shape[0]):
    y_trans_current=[]
    for j in range(m):
        y_trans_current.append(kernel_func(bw,y_list[j],y_test[i][0]))
    y_trans_test.append(y_trans_current)
y_trans_test=np.array(y_trans_test)


# In[11]:


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


# In[12]:





n=x_train.shape[0]
p=x_train.shape[1]
#res_d=1


x_train = torch.tensor(x_train).to(torch.float)
x_test = torch.tensor(x_test).to(torch.float)

y_train = torch.tensor(y_trans_train).to(torch.float)
y_test = torch.tensor(y_trans_test).to(torch.float)



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


# In[13]:


optimizer = torch.optim.Adam(model_nn.parameters(), 
                            lr=0.001,weight_decay=0)
epochs = 100
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
    dcor_test = dcor.distance_correlation(np.float64(y_suff_test.detach().numpy()),np.float64(z_test))

    if epoch % 25 == 24:
        print(f"Epoch: {epoch} | Loss: {loss:.5f} | Test Loss: {loss_test:.5f} | Test Dcor: {dcor_test:.5f}")
        now = datetime.now()
        current_time = now.strftime("%H:%M:%S")
        print("Current Time =", current_time)
model_nn.eval()
with torch.inference_mode():
    y_pred_test, y_suff_test = model_nn(x_test)
#dcor_list.append(dcor_current)
#print(model1, model2,  dcor_current)
#dcor_list_df=pd.DataFrame(dcor_list)
#dcor_list_df.to_csv("./results-BENN-unified-std/result-" + str(model1) + "-" + str(model2) + "-" + str(m) + "-" + str(n) + ".csv")
#print(model1, model2, np.mean(dcor_list), np.std(dcor_list))


# In[14]:


t1_stop = time.time() 
t2_stop = process_time()


# In[15]:


y_suff_test=y_suff_test.numpy()
y_suff_test_df=pd.DataFrame(y_suff_test)
y_suff_test_df.to_csv("./results-BENN-unified-std/result-" + str(model1) + "-" + str(model2) + "-" + str(m) + "-" + str(n) + "/y_suff_" + str(t) + ".csv")


# In[16]:


time_use=[t1_stop-t1_start,t2_stop-t2_start]
time_use_df=pd.DataFrame(time_use)
time_use_df.to_csv("./results-BENN-unified-std/result-" + str(model1) + "-" + str(model2) + "-" + str(m) + "-" + str(n) + "/time_" + str(t) + ".csv")


# In[17]:



#dcor.distance_correlation(np.float64(y_suff_test),np.float64(z_test))


# In[18]:


#plt.plot(abs(y_suff_test[:,0]-0.03*y_suff_test[:,1]**2+0.4)**0.5,
#         y_test-2*x_test[:,0], 'o')


# In[19]:


#dcor.distance_correlation(np.float64(y_suff_test),
#                          np.float64(np.concatenate((z_test.to_numpy()[:,0].reshape((z_test.shape[0],1)),
#                                                     (z_test.to_numpy()[:,1]**2+4*z_test.to_numpy()[:,0]**2).reshape((z_test.shape[0],1))), axis=1)),method="naive")


# In[20]:


#x_test.detach().numpy()[:,1]


# In[21]:


#plt.plot(x_test.detach().numpy()[:,1],y_test, 'o')


# In[22]:


#dcor.distance_correlation(np.float64(y_suff_test),np.float64(z_test.to_numpy()))


# In[23]:


#y_suff_test_all


# In[24]:


#plt.plot(np.float64(y_suff_test[:,0]),np.float64(z_test.to_numpy()[:,0]), 'o')


# In[25]:


#dcor.distance_correlation(np.float64(y_suff_test.detach().numpy()),np.float64(z_test.to_numpy()),method="naive")


# In[26]:


#dcor.distance_correlation(np.float64(z_test.to_numpy()[:,1]),np.float64(z_test.to_numpy()),method="naive")


# In[ ]:





# In[ ]:




