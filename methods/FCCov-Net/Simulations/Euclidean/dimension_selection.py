import time, dcor
import numpy as np
import rpy2.robjects as robjects
from scipy.stats import multivariate_t, cauchy
from sklearn.model_selection import train_test_split
from NSDR import NSDR
from rpy2.robjects import r, numpy2ri
from rpy2.robjects.packages import importr
from support_function import *
import matplotlib.pyplot as plt

def func(x, mode):
    n = x.shape[0]
    sigma = 0.25
    if mode==1:    
        z = np.zeros((n, 3))
        e = np.random.normal(0, sigma, (n,3))

        z[:,0] = x[:,0]**2/(1+(0.1+0.5*x[:,1])**2) + e[:,0]
        z[:,1] = e[:,1]
        z[:,2] = e[:,2]

    elif mode==4:
        z = np.zeros((n, 1))
        e = np.random.normal(0, sigma, (n,1))
        z[:,0] = np.abs(x[:,0])**0.5 + x[:,0]/(x[:,1]+2.5) + e[:,0]

    elif mode==10:    # outlier test
        sigma=0.25
        z = np.zeros((n, 3))
        e = np.random.normal(0, sigma, (n,3))
        r = 0.1
        indices_to_replace = np.random.choice(n, size=int(n * r), replace=False) 
        outlier = np.random.uniform(-50, 50, (n,1))
        e[indices_to_replace, 0] = outlier[indices_to_replace, 0]
        z[:,0] = x[:,0]**2/(1+(0.1+0.5*x[:,1])**2) + e[:,0]
        z[:,1] = e[:,1]
        z[:,2] = e[:,2]
    
    return z

def func_1(x, mode):
    n = x.shape[0]
    if mode==1 or mode==10:
        z = np.zeros((n, 1))
        z[:,0] = x[:,0]**2/(1+(0.1+0.5*x[:,1])**2)

    elif mode==4:
        z = np.zeros((n, 1))
        z[:,0] = np.abs(x[:,0])**0.5 + x[:,0]/(x[:,1]+2.5)

    return z

embedding = func_1

def data_generator_1(n, p, func_mode, x_mode):
    test_size = int(0.2*n)
    n = n + test_size  # add extra test samples on top of the original data
    x = x_generator(n, p, x_mode)
    y = func(x,func_mode)
    return train_test_split(x, y, test_size=test_size)

def x_generator(n, p, mode):
    mean = np.zeros(p)
    cov = np.zeros((p, p))
    for i in range(p):
        for j in range(p):
            cov[i, j] = 0.5 ** abs(i - j)

    if mode==0:          # represents the standard normal distribution
        x = np.random.normal(0, 1, (n,p))

    elif mode==1:        # represents a non-standard normal distribution
        x = np.random.multivariate_normal(mean, cov, n)

    elif mode==3:
        low = -2
        high = 2
        x = np.random.uniform(low, high, (n, p))

    elif mode==10:      # used for outlier testing
        r = 0.5
        x = np.random.multivariate_normal(mean, cov, n)
        cauchy_data = cauchy.rvs(size=(n, 3)) * 2    # generate three-dimensional Cauchy samples
        # randomly choose the indices to replace
        indices_to_replace = np.random.choice(n, size=int(n * r), replace=False)

        # replace the data
        x[indices_to_replace, 0] = cauchy_data[indices_to_replace, 0]    # replace X_1
        
    return x

# Compute cumulative matrix
def dist(y1,y2,mode=0):          # y1 may be a matrix, y2 must be a vector
    if mode==0:                  # 0 denotes Euclidean distance (default)
        return np.linalg.norm(y1-y2, axis=1)
    if mode==1:                  # 1 denotes spherical distance
        return np.arccos(np.dot(y1,y2))
    if mode==3:                  # Hellinger distance
        return np.sqrt(np.sum((np.sqrt(y1) - np.sqrt(y2))**2
                              , axis=1)) / np.sqrt(2)
    if mode==4:                  # TV distance
        return 0.5 * np.sum(np.abs(y1 - y2))

# Means are left intact; centering happens externally
def calculate_distances_and_sort(X, Y, mode):
    X_combined = []
    n = Y.shape[0]
    if X.ndim == 1:
        for i in range(n):
            # Compute distances from every sample to the i-th sample
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
    # Mean-centering
    X = X - np.mean(X, axis=0).reshape(1,-1)
    n, d = X.shape
    # Build the MCCOV matrix
    mcov_ma = np.zeros((d,d))
    for i in range(d):
        for j in range(d):
            mcov_ma[i,j] = mcov(X[:,i], X[:,j], Y, 0)
    return mcov_ma

def cumulative_explained_variance(matrix):
    # Compute eigenvalues and eigenvectors
    eigenvalues, _ = np.linalg.eig(matrix)
    
    eigenvalues = np.abs(eigenvalues)
    # print(eigenvalues)
    # Sort eigenvalues in descending order
    sorted_eigenvalues = np.sort(eigenvalues)[::-1]
    
    # Compute explained variance per eigenvalue
    explained_variances = sorted_eigenvalues / np.sum(sorted_eigenvalues)
    
    # Compute cumulative explained variance
    cumulative_explained_variance = np.cumsum(explained_variances)
    
    return (cumulative_explained_variance, explained_variances)

def main():
    dist_mode = 0
    categorical_y = False
    outlier = False
    device = 'cuda:1'
    func_mode = 1
    x_mode = 0
    p_dim = 10
    n_sample = 1000
    d = p_dim

    rep = 10
    target = []
    threshold = 0.95
    for i in range(rep):
        x_train, x_test, y_train, y_test = data_generator_1(n=n_sample, p=p_dim, func_mode=func_mode, x_mode=x_mode)
        model = NSDR(neural_network=NSDR().generate_default_net(p_dim, output_dim=d), max_dim=1, method = "fnormsquare", adaptive_cv=False, retrain = True, 
                    early_stop=False ,debug=True, device=device, initial_blambda = np.sqrt(d) * 1e-3,dist_mode = dist_mode 
                    ,loss_mode='mccov', iter_num=100, categorical_y=categorical_y)
        model.fit(x_train, y_train)
        dr_train = model.transform(x_train)
        accum = mcov_matrix(dr_train, y_train, dist_mode)
        
        target.append(np.argmax(cumulative_explained_variance(accum)[0]>threshold)+1)

    # Histogram bin edges
    bins = [0.7, 1.3, 1.7, 2.3, 2.7, 3.3]

    # Plot histogram via matplotlib
    plt.hist(target, bins=bins, alpha=0.7, color='blue', edgecolor='black', rwidth=0.8)

    # Set horizontal axis limits
    plt.xlim(0, 4)

    # Use integer ticks from 1 to 5
    plt.xticks(np.arange(1, 6, 1))

    # Add plot title and axis labels
    plt.title('Histogram of target dimension for model II')
    plt.xlabel('Target Dimention')
    plt.ylabel('Frequency')
    plt.savefig('dimension decision_1.png')
    # Show the figure
    plt.show()

if __name__ == "__main__":
    main()
