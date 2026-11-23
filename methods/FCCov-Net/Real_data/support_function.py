import numpy as np
import dcor
import torch.nn as nn
from scipy.stats import norm
from scipy.spatial import distance
from sklearn.metrics.pairwise import rbf_kernel
from scipy.spatial.distance import pdist, squareform

from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.multioutput import MultiOutputClassifier
from sklearn.neighbors import KNeighborsRegressor
from scipy.optimize import minimize

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, TensorDataset
import time
import pandas as pd

def spd_dis(A,B):
    matrix_ABA = torch.matmul(torch.matmul(tensor_power(A,-0.5),B),tensor_power(A,-0.5))
    S,U = torch.linalg.eigh(matrix_ABA)
    S_trans = (torch.log(S))**2

    dis = torch.sum(S_trans, dim=-1)
    return torch.sqrt(dis)

def spd_disL(A, B):
    L_1 = torch.linalg.cholesky(A)
    L_2 = torch.linalg.cholesky(B)
    D_1 = torch.diagonal(L_1, dim1=-2, dim2=-1)
    D_2 = torch.diagonal(L_2, dim1=-2, dim2=-1)
    P_1 = torch.tril(L_1)-D_1
    P_2 = torch.tril(L_2)-D_2
    
    dis = 0
    dis += torch.norm(P_1-P_2, p='fro', dim=(-2,-1))**2 
    dis += torch.sum((torch.log(D_1)-torch.log(D_2))**2, dim=-1)
    return torch.sqrt(dis)

def spd_plus(A,B):
    matrix_plus = expm(logm(A) + logm(B))

    return matrix_plus

def spd_minus(A,B):
    matrix_minus = expm(logm(A) - logm(B))

    return matrix_minus

def spd_mul(A,r):
    matrix_mul = expm(r * logm(A))

    return matrix_mul

def Log(Y,Z):
    log_YZ = logm(torch.matmul(torch.matmul(tensor_power(Y,-0.5),Z),tensor_power(Y,-0.5)))
    result = torch.matmul(torch.matmul(tensor_power(Y,0.5),log_YZ),tensor_power(Y,0.5))

    return result

def logm(Y):
    S, U = torch.linalg.eigh(Y)
    log_S = torch.log(S)
    log_Y = torch.matmul(torch.matmul(U,torch.diag_embed(log_S)),U.transpose(1,2))
    return log_Y

def Exp(Y,Z):
    exp_YZ = expm(torch.matmul(torch.matmul(tensor_power(Y,-0.5),Z),tensor_power(Y,-0.5)))
    result = torch.matmul(torch.matmul(tensor_power(Y,0.5),exp_YZ),tensor_power(Y,0.5))

    return result

def expm(Y):
    S, U = torch.linalg.eigh(Y)
    exp_S = torch.exp(S)
    exp_Y = torch.matmul(torch.matmul(U,torch.diag_embed(exp_S)),U.transpose(1,2))
    return exp_Y

def tensor_power(A,r): # Generally a 3D tensor, the first dimension can be 1
    S, U = torch.linalg.eigh(A)
    pow_S = S**r
    power_A = torch.matmul(torch.matmul(U,torch.diag_embed(pow_S)),U.transpose(1,2))

    return(power_A)

def generate_init(m):       # Generate a positive definite matrix by controlling the minimum eigenvalue of a symmetric matrix
    A = np.random.rand(m, m)
    A = 0.5 * (A + A.T)
    eigenvalues, _ = np.linalg.eig(A)
    A += np.eye(m) * (np.abs(np.min(eigenvalues)) + 1e-8)
    return A

def rand_matrix(d):        # Generate random matrices with d=2 or d=3
    A = np.zeros((d,d))
    for i in range(d):
        for j in range(i+1,d):
            A[i,j] = np.random.normal(0,0.5)
    A = A + A.T
    # Generate diagonal elements following N(0, 1)
    diagonal = np.random.normal(0, 1, d)
    # Fill the diagonal elements into the matrix
    np.fill_diagonal(A, diagonal)
    return torch.tensor(A)

def generate_rm(n,d):       # Generate n random matrices
    matrices = [rand_matrix(d) for _ in range(n)]
    # Convert list to Tensor with shape [n, d, d]
    return torch.stack(matrices)
    

def d2_Nmm(off_diag, sigma):    # Given a mean D, generate the corresponding Nmm
    n = off_diag.shape[0]
    off_diag = off_diag.squeeze()
    matrices = []
    for i in range(n):
        D = torch.tensor([[1, off_diag[i]], [off_diag[i], 1]])
        A = sigma*rand_matrix(2) + D
        matrices.append(A)
    return torch.stack(matrices)

def d3_Nmm(off_diag, sigma):
    n, d = off_diag.shape
    matrices = []
    for i in range(n):
        D = torch.tensor([[1, off_diag[i, 0], off_diag[i, 1]], 
                        [off_diag[i, 0], 1, off_diag[i, 0]],
                        [off_diag[i, 1], off_diag[i, 0], 1]])
        A = sigma*rand_matrix(3) + D
        matrices.append(A)
    return torch.stack(matrices)

def distribution_nor(rho_1, rho_2, sigma):   # rho_1, rho_2 are both scalar functions of X, n-dimensional vectors when there are n samples
    n = rho_1.shape[0]
    Y = []
    for i in range(n):
        mean = rho_1[i]
        mean_Y = np.random.normal(mean, sigma)
        var_Y = rho_2[i]  # Calculate the variance of Y
        Y.append(np.random.normal(mean_Y, var_Y))
    return np.array(Y)

def quantile_nor(mean, sigma, var_Y):          # var_Y needs to be a column vector
    mean_Y = np.random.normal(mean, sigma)     # If there are n samples, generate n corresponding normal mean_Y
    quant = norm.ppf(np.linspace(0.001, 0.999, 21))      # Generate 21 quantiles of the standard normal distribution
    tile_1 = np.tile(mean_Y.reshape((-1,1)), (1,21))
    tile_2 = var_Y*quant
    return (tile_1 + tile_2)

# Distances between distributions for real data

def hellinger_distance(p, q):
    """Calculate the Hellinger distance between two probability distributions"""
    return np.sqrt(np.sum((np.sqrt(p) - np.sqrt(q))**2)) / np.sqrt(2)

def total_variation_distance(p, q):
    """Calculate the total variation distance (TV distance) between two probability distributions"""
    return 0.5 * np.sum(np.abs(p - q))

def pairwise_distance_matrix(Y, distance_func):
    """Calculate the pairwise distance matrix for all row pairs using the specified distance function"""
    n = Y.shape[0]
    D = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            D[i, j] = distance_func(Y[i], Y[j])
            D[j, i] = D[i, j]
    return D

def dcor_metric(x, y, mode):
    # Calculate distance matrix
    A = distance.squareform(distance.pdist(x, 'euclidean'))
    if mode==3:
        B = pairwise_distance_matrix(y, hellinger_distance)
    elif mode==4:
        B = pairwise_distance_matrix(y, total_variation_distance)

    # Double-centered matrices
    A = dcor.double_centered(A)
    B = dcor.double_centered(B)
    
    return dcor.distance_correlation(A, B)

# Functions needed for WIRE comparison in simulations
def matpower(a, alpha):
    a = 0.5*(a+a.T)
    val, vec = np.linalg.eigh(a)
    return vec@np.diag(val**alpha)@vec.T

def sym(a):
    return 0.5*(a+a.T)

def onorm(a):
    val, vec = np.linalg.eigh(a)
    return np.max(val)

# Performance is concerning
def gram_gauss(x, x_t, complexity):
    n = x.shape[0]
    m = x_t.shape[0]
    k2 = x@x.T
    k1 = np.tile(np.diag(k2), (n,1)).T
    k3 = k1.T
    k = k1 - 2*k2 + k3
    sigma = np.sum(np.sqrt(k))/(n*(n-1))
    gamma = complexity/(2*sigma**2)
    ma_1 = np.tile(np.diag(x@x.T), (m, 1)).T
    ma_2 = x@x_t.T
    ma_3 = np.tile(np.diag(x_t@x_t.T), (n, 1))
    return np.exp(-gamma*(ma_1-2*ma_2+ma_3))

def wire(x_train, x_test, dis_mat, d, eps, gamma):
    n = x_train.shape[0]
    Q = np.identity(n) - np.ones((n,n))/n
    Kx = gram_gauss(x_train, x_train, gamma)
    Gx = Q@Kx@Q
    Gxinv = matpower(sym(Gx+eps*onorm(Gx)*np.identity(n)), -1)

    a = Gxinv@Gx
    b = dis_mat@Gx@dis_mat
    coef_M = a@b@a.T
    val, vec = np.linalg.eigh(coef_M)
    # Select the largest d eigenvectors
    sub_vec = vec[:,-d:]
    # Dimension reduction function for training set
    f_train = (sub_vec.T@Gxinv@Q@Kx).T

    Kx_t = gram_gauss(x_train, x_test, gamma)
    f_test = (sub_vec.T@Gxinv@Q@Kx_t).T
    return (f_train, f_test)

def wire_gcv(x,y,eps,complex):
    n, p = x.shape
    Kx = gram_gauss(x, x, complex)
    Ky = y
    G1 = Ky@Kx
    G2 = Kx
    G2inv = matpower(G2+eps*onorm(G2)*np.identity(n),-1)
    nu = np.sum((G1-G2@G2inv@G1)**2)
    de = (1- np.sum(np.diag(G2inv@G2))/n)**2
    return (nu/de)

# Default values for cross-validation
# eps = 0.01
# gamma = 0.2
def param(x_train, D, cv):
    if cv==True:
        row = 6
        col = 9
        rho = [1/5,1/4,1/3,1/2,1,2,3,4,5]
        params = np.zeros((row,col))
        for i in range(row):
            for j in range(col):
                params[i,j] = wire_gcv(x_train,D,10**(-(i+1)), rho[j])
        k,l = np.where(params == np.min(params))
        ex = 10**(-k.item())
        gamma = rho[l.item()]
        return (ex, gamma)
    else:
        return (0.01, 0.2)

# Distance matrix required for WIRE
def distance_matrix(y, mode):
    if mode==0:   # Euclidean distance for Euclidean vectors
        diff = y[:, np.newaxis, :] - y[np.newaxis, :, :]
        D = np.sqrt(np.sum(diff**2, axis=-1))
    if mode==1:   # Spherical distance
        temp = y@(y.T)
        ma = np.maximum(-1, np.minimum(1, temp))
        D = np.arccos(ma)
        np.fill_diagonal(D,0)
    if mode==2:   # SPD distance
        n = y.shape[0]
        p = y.shape[1]
        D = np.zeros((n,n))
        for i in range(n):
            for j in range(n):
                if i<j:
                    D[i,j] = spd_dis(y[i].reshape((1,p,p)),y[j].reshape((1,p,p))).item()
                D[j,i] = D[i,j]
    if mode==3:   # Hellinger distance
        pairwise_dists = pdist(y, metric=hellinger_distance)
        D = squareform(pairwise_dists)
    if mode==4:    # Total Variation distance
        pairwise_dists = pdist(y, metric=total_variation_distance)
        D = squareform(pairwise_dists)
    return D


# Gaussian kernel function
def gaussian_kernel(x, x_i, bandwidth):
    return np.exp(-0.5 * np.linalg.norm(x - x_i)**2 / bandwidth**2)

# Weighted Hellinger distance
def weighted_hellinger(w, X, Y, x, bandwidth):
    n = X.shape[0]
    weighted_distance = 0
    for i in range(n):
        kernel_weight = gaussian_kernel(x, X[i], bandwidth)
        weighted_distance += kernel_weight * hellinger_distance(Y[i], w)
    return weighted_distance

# Define objective function
def objective(w, X, Y, x, bandwidth):
    return weighted_hellinger(w, X, Y, x, bandwidth)


# Local NW kernel frechet regression with Hellinger distance
# k represents the local KNN parameter
def loc_frechet(train_data, train_label, test_data, bandwidth=1.0, k=4):
    constraints = ({
    'type': 'eq',
    'fun': lambda w: np.sum(w) - 1
    })
    # Define boundary conditions
    bounds = [(0, 1) for _ in range(6)]
    # Initial guess
    w0 = np.full(6, 1/6)
    pred_label = []
    if len(test_data.shape) ==1:
        test_data = test_data.reshape(1,-1)
    for x in test_data:
        # Find the k nearest points and corresponding labels
        distances = np.linalg.norm(train_data - x, axis=1)
        nearest_indices = np.argsort(distances)[:k]
        nearest_X = train_data[nearest_indices]
        nearest_Y = train_label[nearest_indices]
        
        result = minimize(objective, w0, args=(nearest_X, nearest_Y, x, bandwidth), 
                          constraints=constraints, bounds=bounds)
        pred_label.append(result.x)
    return np.array(pred_label)

# Define LDL model
class LDLModel(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(LDLModel, self).__init__()
        self.layer = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Linear(32,output_dim),
            nn.Softmax(dim=1)
        )
    
    def forward(self, x):
        return self.layer(x)
    
    def transform(self, test_data):
        
        with torch.no_grad():
            if not isinstance(test_data, torch.Tensor):
                test_data = torch.tensor(test_data, dtype=torch.float32)

            # Concatenate outputs from all batches
            output = self.forward(test_data)

            if not isinstance(output, np.ndarray):
                # If not, try to convert to numpy.ndarray
                if isinstance(output, torch.Tensor):
                    # Ensure PyTorch tensor is moved to CPU and converted to NumPy array
                    output = output.cpu().numpy()
                else:
                    output = np.array(output)

        return output.astype(np.float32)
    
    
def divergence(dis_1, dis_2, mode=0):
    n = dis_1.shape[0]
    if n != dis_2.shape[0]:
        print("dimention dismissing")
        return

    distance = 0
    if mode==0:  # Euclidean distance
        temp = np.linalg.norm((dis_1-dis_2), axis=1)
        distance = np.mean(temp)
    elif mode==1:  # KL-divergence
        num = 0
        for i in range(n):
            num += np.sum(dis_1[i]*np.log(dis_1[i]/dis_2[i]))
        distance = num/n
    elif mode==2: # Sorensen distance
        num = 0
        for i in range(n):
            num += np.sum(np.abs(dis_1-dis_2))/np.sum(dis_1+dis_2)
        distance = num/n
    elif mode==3: # Squared Chi-square distance
        num = 0
        for i in range(n):
            num += np.sum((dis_1-dis_2)**2/(dis_1+dis_2))
        distance = num/n
        
    return distance

def softmax(x):
    x -= np.max(x, axis=1, keepdims=True)
    x = np.exp(x)/ np.sum(np.exp(x), axis=1, keepdims=True)
    return x

def LDL_BP(x_train, x_test, train_res, test_res, tr_dis, te_dis, d, num_epochs=100, batch_size = 32):
    num_epochs = num_epochs
    batch_size = batch_size
    p_dims = d
    num_labels = 6
    
    model = LDLModel(input_dim=p_dims, output_dim=num_labels)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    for epoch in range(num_epochs):
        permutation = np.random.permutation(x_train.shape[0])
    
        for i in range(0, x_train.shape[0], batch_size):
            indices = permutation[i:i + batch_size]
            batch_X, batch_y = x_train[indices], tr_dis[indices]

            # Convert NumPy arrays to PyTorch tensors
            batch_X = torch.tensor(batch_X, dtype=torch.float32)
            batch_y = torch.tensor(batch_y, dtype=torch.float32)

            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
    
    model_wire = LDLModel(input_dim=p_dims, output_dim=num_labels)
    optimizer = optim.Adam(model_wire.parameters(), lr=0.001)
    for epoch in range(num_epochs):
        permutation = np.random.permutation(train_res.shape[0])
    
        for i in range(0, train_res.shape[0], batch_size):
            indices = permutation[i:i + batch_size]
            batch_X, batch_y = train_res[indices], tr_dis[indices]

            # Convert NumPy arrays to PyTorch tensors
            batch_X = torch.tensor(batch_X, dtype=torch.float32)
            batch_y = torch.tensor(batch_y, dtype=torch.float32)

            optimizer.zero_grad()
            outputs = model_wire(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
    
    test_pred = model.transform(x_test)
    te_pred = model_wire.transform(test_res)
    
    result = np.zeros((2,4))
    for i in range(4):
        result[0, i] = divergence(test_pred, te_dis, i)
        result[1, i] = divergence(te_pred, te_dis, i)
        
    return result

def LDL_KNN(x_train, x_test, train_res, test_res, tr_dis, te_dis, k = 6, bandwidth = 20):
    k = k
    bandwidth = bandwidth

    m_te_pred = loc_frechet(x_train, tr_dis, x_test, bandwidth, k)
    w_te_pred = loc_frechet(train_res, tr_dis, test_res, bandwidth, k)
    
    result = np.zeros((2,4))
    for i in range(4):
        result[0,i] = divergence(m_te_pred, te_dis, i)
        result[1,i] = divergence(w_te_pred, te_dis, i)
    return result

def classification(x_train, x_test, train_res, test_res, y_train, y_test):
    acc = np.zeros((2,4))
    
    for i in range(2):
        if i==0:
            train_data = x_train
            test_data = x_test
        elif i==1:
            train_data = train_res
            test_data = test_res
        
        # scaler = StandardScaler()
        # pipeline = make_pipeline(scaler, LogisticRegression(multi_class="multinomial", max_iter=200))
        # pipeline.fit(train_data, y_train)
        # acc[i,0] = accuracy_score(y_test, pipeline.predict(test_data))
        lrflenet = LogisticRegression(multi_class="multinomial", max_iter=1000, solver='saga')
        lrflenet.fit(train_data, y_train)
        acc[i,0] = accuracy_score(y_test, lrflenet.predict(test_data))
        
        knn = KNeighborsClassifier(n_neighbors=6)
        knn.fit(train_data, y_train)
        acc[i,1] = accuracy_score(y_test, knn.predict(test_data))
        
        ran_for = RandomForestClassifier(n_estimators=100, random_state=41)
        ran_for.fit(train_data, y_train)
        acc[i,2] = accuracy_score(y_test, ran_for.predict(test_data))
        
        svm = SVC(kernel='linear')
        svm.fit(train_data, y_train)
        acc[i,3] = accuracy_score(y_test, svm.predict(test_data))
    return acc


# Calculate cumulative matrix
def dist(y1,y2,mode=0):          # y1 can be a matrix, while y2 can only be a vector
    if mode==0:                  # 0 for Euclidean distance, default is Euclidean
        return np.linalg.norm(y1-y2, axis=1)
    if mode==1:                  # 1 for spherical distance
        return np.arccos(np.dot(y1,y2))
    if mode==3:                  # Hellinger distance
        return np.sqrt(np.sum((np.sqrt(y1) - np.sqrt(y2))**2
                              , axis=1)) / np.sqrt(2)
    if mode==4:                  # TV distance
        return 0.5 * np.sum(np.abs(y1 - y2))

# Do not subtract mean, mean subtraction is done externally
def calculate_distances_and_sort(X, Y, mode):
    X_combined = []
    n = Y.shape[0]
    if X.ndim == 1:
        for i in range(n):
            # Calculate the distance from each sample to the i-th sample
            distances = dist(Y,Y[i],mode)
            sorted_indices = np.argsort(distances)
            sorted_X = X[sorted_indices]
            X_combined.append(sorted_X)
    
    elif X.ndim == 2:
        mean_X = np.mean(X, axis=0)
        for i in range(n):
            distances = dist(Y,Y[i],mode)
            sorted_indices = np.argsort(distances)
            sorted_X = X[sorted_indices]
            X_combined.append(sorted_X)

    return np.array(X_combined)

def mcov(f_a, f_b, Y, mode):
    n = Y.shape[0]
    sort_a = calculate_distances_and_sort(f_a, Y, mode)
    sort_b = calculate_distances_and_sort(f_b, Y, mode)
    T=0
    
    cumu_a = np.cumsum(sort_a, axis=1)
    cumu_b = np.cumsum(sort_b, axis=1)
    T += np.sum((cumu_a*cumu_b)[:,:n-1])
    T -= np.sum(np.cumsum(sort_a*sort_b, axis=1)[:,:n-1])
    T -= np.sum(np.cumsum(f_a.reshape(-1,1)*sort_b + f_b.reshape(-1,1)*sort_a, axis=1)[:,:n-1])
    
    T += 2*(n-1)*np.sum(f_a*f_b)
    return T/(n*(n-1)*(n-2)*(n-3))

def mcov_matrix(X, Y, mode=0):
    # Demean
    X = X - np.mean(X, axis=0).reshape(1,-1)
    n, d = X.shape
    # Create MCCOV matrix
    mcov_ma = np.zeros((d,d))
    for i in range(d):
        for j in range(d):
            mcov_ma[i,j] = mcov(X[:,i], X[:,j], Y, 0)
    return mcov_ma