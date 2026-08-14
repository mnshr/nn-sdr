#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch, time, dcor, copy, scipy, os
import numpy as np
#import rpy2.robjects as robjects
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal, multivariate_t, uniform, beta, poisson
from torch import nn
from scipy.spatial.distance import cdist
from sklearn.model_selection import train_test_split
from scipy.linalg import eig
from NSDR import NSDR
from plotutils import cum_plot, prop_plot
#from rpy2.robjects import r, numpy2ri
from sklearn.preprocessing import OneHotEncoder
#from rpy2.robjects.packages import importr
#importr("nsdr")
#%load_ext rpy2.ipython
import argparse
import pandas as pd
from time import process_time

from pytorch_lightning import seed_everything
seed_everything(42, workers=True)
torch.use_deterministic_algorithms(True, warn_only=True)


# In[2]:


def sigmoid(x):
    return 1 / (1 + np.exp(-x))



    return train_test_split(x, y, test_size=test_size)
                
def init_xv_uniform(m):
    if type(m) == nn.Linear:
        nn.init.xavier_uniform_(m.weight)
        nn.init.zeros_(m.bias)
    
def index_convertion(max_num_list, current_list):
    max_num = list(max_num_list)
    current = list(current_list)
    assert len(max_num) == len(current)
    idx = 0
    prod = 1
    for i in list(range(len(max_num)))[::-1]: ## reversee the oder
        idx = idx + current_list[i] * prod
        prod = prod * max_num_list[i]
    return idx

    


# In[3]:


#rep_num = 100
#y_mode_num = 6 ##number of generalized distance
#n = 1000 # training size + test size
#p = 50
#d = 2


iter_num = 100
batch_size = 100 # p * 10


# In[4]:


parser = argparse.ArgumentParser(description="Running GMDD")
parser.add_argument('--model1', default=4, type = int, help = 'model1')
parser.add_argument('--model2', default=1, type = int, help = 'model2')
parser.add_argument('--n', default=1000, type = int, help = 'n')
parser.add_argument('--d', default=2, type = int, help = 'd')
parser.add_argument('--t', default=3, type = int, help = 't')
args = parser.parse_args()
model1 = args.model1
model2 = args.model2
n = args.n
res_d = args.d
t = args.t
print(model1, model2, n, res_d, t)


# In[19]:


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


# In[20]:


directory="./results-GMDD-correct/result-" + str(model1) + "-" + str(model2) + "-" + str(n)
if not os.path.exists(directory):
    os.makedirs(directory)


# In[21]:


x_train=x_train.to_numpy()
y_train=y_train.to_numpy()
x_test=x_test.to_numpy()
y_test=y_test.to_numpy()
z_test=z_test.to_numpy()
n=x_train.shape[0]
p=x_train.shape[1]


# In[22]:


t1_start = time.time() 
t2_start = process_time()


# In[23]:



net_seq = NSDR().generate_default_net(p)
net_seq.apply(init_xv_uniform)

model = NSDR(neural_network=net_seq, max_dim=res_d, method = "seq", adaptive_cv=False, 
             retrain = True, early_stop=False ,debug=False, device="cpu", 
             categorical_y = False, iter_num=iter_num, y_mode=0)
start_time = time.time()
model.fit(x_train, y_train)
y_suff_test=model.transform(x_test)


# In[ ]:



t1_stop = time.time() 
t2_stop = process_time()


# In[ ]:



y_suff_test_df=pd.DataFrame(y_suff_test)
y_suff_test_df.to_csv("./results-GMDD-correct/result-" + str(model1) + "-" + str(model2)  + "-" + str(n) + "/y_suff_" + str(t) + ".csv")


# In[ ]:


time_use=[t1_stop-t1_start,t2_stop-t2_start]
time_use_df=pd.DataFrame(time_use)
time_use_df.to_csv("./results-GMDD-correct/result-" + str(model1) + "-" + str(model2) + "-" + str(n) + "/time_" + str(t) + ".csv")


# In[ ]:





# In[ ]:




