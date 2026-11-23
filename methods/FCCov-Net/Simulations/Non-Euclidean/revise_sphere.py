import os
import torch, time, dcor, os
import numpy as np
from scipy.stats import multivariate_t
from support_function import *
from mynet import *
from joblib import Parallel, delayed


# Spherical data generation
def func(x,mode):
    n = x.shape[0]
    sigma = 0.2
    if mode==0:           # Indicates 2D sphere
        f_1 = 0.25*x[:,0]+0.75*x[:,1]
        f_2 = 0.75*x[:,-2]+0.25*x[:,-1]
        z = np.zeros((n, 3))
        e = np.random.normal(0, sigma,(n,2))

        z[:,0] = np.sqrt(1-f_1**2)*np.cos(f_2)
        z[:,1] = np.sqrt(1-f_1**2)*np.sin(f_2)
        z[:,2] = f_1**2
        for i in range(n):
            A = np.linalg.svd(np.eye(3) - z[i].reshape(-1, 1) @ z[i].reshape(1, -1),
                              full_matrices=False)[0][:, :2]
            epsilon = A @ e[i]
            norm_epsilon = np.linalg.norm(epsilon)
            z[i] = np.cos(norm_epsilon)*z[i] + np.sin(norm_epsilon) * epsilon / norm_epsilon

    elif mode==1:         # Indicates 3D sphere
        f_1 = np.linalg.norm(x[:,:2],axis=1)
        f_2 = np.linalg.norm(x[:,-2:],axis=1)
        # f_1 = x[:,0] + x[:,1]
        z = np.zeros((n,3))
        e = np.random.normal(0, sigma, (n,2))

        z[:,0] = np.sin(f_1+e[:,0])*np.sin(f_2+e[:,1])
        z[:,1] = np.sin(f_1+e[:,0])*np.cos(f_2+e[:,1])
        z[:,2] = np.cos(f_1+e[:,0])

    return z

def func_1(x, mode):
    n = x.shape[0]

    if mode==0:
        f_1 = 0.25*x[:,0]+0.75*x[:,1]
        f_2 = 0.75*x[:,-2]+0.25*x[:,-1]
        z = np.zeros((n, 2))
        z[:,0] = f_1
        z[:,1] = f_2
    elif mode==1:
        f_1 = np.linalg.norm(x[:,:2],axis=1)
        f_2 = np.linalg.norm(x[:,-2:],axis=1)
        z = np.zeros((n, 2))
        z[:,0] = f_1
        z[:,1] = f_2
    return z

embedding = func_1

def x_generator(n, p, mode):
    mean = np.zeros(p)
    cov = np.zeros((p, p))
    for i in range(p):
        for j in range(p):
            cov[i, j] = 0.5 ** abs(i - j)

    if mode==-1:         # Indicates d-dimensional [0,1] uniform distribution
        x = np.random.uniform(0, 1, (n,p))
    if mode==0:          # Indicates standard normal distribution
        x = np.random.normal(0, 1, (n,p))
    elif mode==1:        # Indicates non-standard normal distribution
        x = np.random.multivariate_normal(mean, cov, n)
    elif mode==2:        # Multivariate t-distribution with 2 degrees of freedom
        df = 2
        mul_t = multivariate_t(mean, cov, df)
        x = mul_t.rvs(n)
    elif mode==3:
        low = -2
        high = 2
        x = np.random.uniform(low, high, (n, p))
    elif mode==4:
        x = 0.5*np.random.multivariate_normal(np.ones(p), np.identity(p), n) + 0.5*np.random.multivariate_normal(-1*np.ones(p), np.identity(p), n)
    elif mode==5:
        x = np.random.multivariate_normal(mean, 0.6*np.identity(p)+0.41*np.ones((p,p)), n)

    return x


def data_generator(n, p, func_mode, x_mode):    
    test_size =  int(0.2*n)                     
    x = x_generator(n, p, x_mode)
    y = func(x,func_mode)
    x_t = x_generator(test_size, p, x_mode)    
    y_t = func(x_t, func_mode)
    return (x,x_t,y,y_t)

# Distance matrix needed for WIRE
def distance_matrix(y, mode, device):
    if mode==0:   # Euclidean distance for Euclidean vectors
        diff = y[:, np.newaxis, :] - y[np.newaxis, :, :]
        D = np.sqrt(np.sum(diff**2, axis=-1))
    if mode==1:   # Spherical distance
        temp = y@(y.T)
        ma = np.maximum(-1, np.minimum(1, temp))
        D = np.arccos(ma)
        np.fill_diagonal(D,0)
    if mode==2:   # SPD affine invariant distance
        y = y.to(device)
        n = y.shape[0]
        p = y.shape[1]
        D = np.zeros((n,n))
        for i in range(n):
            for j in range(n):
                if i<j:
                    D[i,j] = spd_dis(y[i].reshape((1,p,p)),y[j].reshape((1,p,p))).item()
                D[j,i] = D[i,j]
    if mode==6:  # SPD log cholesky distance
        y = y.to(device)
        n = y.shape[0]
        p = y.shape[1]
        D = np.zeros((n,n))
        for i in range(n):
            for j in range(n):
                if i<j:
                    D[i,j] = spd_disL(y[i].reshape((1,p,p)),y[j].reshape((1,p,p))).item()
                D[j,i] = D[i,j]
    return D

def param_grid(x_train, x_test, y_train, y_test, device, cv_mccov, arch, iter_num, batch_size, dist_mode, d, func_mode):
    p = x_train.shape[1]
    if cv_mccov==True:
        blambda_values = [1e-5, 1e-4, 5e-4, 1e-3, 0.01]
        performance = []
        y_dcorfnorms = 0
        for bl in blambda_values:
            model = CustomNet(architecture=arch, optimizer_name='Adam', iter_num=iter_num, p_dim=p, batch_size=batch_size,
                              blambda=bl, dist_mode=dist_mode, device=device, output_dim=d)
            model.train_step(x_train, y_train, model.mccov_loss)
            y_hat = model.transform(x_test).astype(np.float64)
            y_dcorfnorms = dcor.distance_correlation(y_hat, embedding(x_test, mode=func_mode))
            performance.append((bl, y_dcorfnorms))
        
        # Return the tuple of optimal hyperparameters and performance
        best_bl, best_per = max(performance, key=lambda x: x[1])
        print("best parameters:",best_bl)
        return (best_bl, best_per)
    else:
        bl=1e-6
        model = CustomNet(architecture=arch, optimizer_name='Adam', iter_num=iter_num, p_dim=p, batch_size=batch_size,
                          blambda=bl, dist_mode=dist_mode, device=device, output_dim=d)
        model.train_step(x_train, y_train, model.mccov_loss)
        y_hat = model.transform(x_test).astype(np.float64)
        y_dcorfnorms = dcor.distance_correlation(y_hat, embedding(x_test, mode=func_mode))
        return (bl, y_dcorfnorms)

def run_repetition(n_sample, p_dim, func_mode, x_mode, loss_modes, dist_mode, device, d, cv_wire, cv_mccov, arch, iter_num, batch_size):
    """
    Runs a single repetition of the simulation.
    This function is designed to be called in parallel.
    """
    # Generate data
    x_train, x_test, y_train, y_test = data_generator(n=n_sample, p=p_dim, func_mode=func_mode, x_mode=x_mode)

    loss_m_val = 0.0
    loss_w_val = 0.0
    mccov_time = 0.0
    wire_time = 0.0

    for loss_mode in loss_modes:
        start_time = time.time()
        if loss_mode == 'wire':
            print("WIRE training start")
            D = distance_matrix(y_train, dist_mode, device) # Calculate distance matrix
            # WIRE parameter cross-validation
            # eps, gamma = param(x_train, D, cv_wire)
            eps = 0.001
            gamma = 0.2
            train_res, test_res = wire(x_train, x_test, D, d, eps, gamma)
            test_dcor = dcor.distance_correlation(test_res, embedding(x_test, mode=func_mode))
            # train_dcor = dcor.distance_correlation(train_res, embedding(x_train, mode=func_mode))
            loss_w_val += test_dcor
            current_wire_time = time.time() - start_time
            wire_time += current_wire_time
            print(f"WIRE training end: {current_wire_time:.2f}s")
        elif loss_mode == 'mccov':
            print("MCCOV training start")
            best_bl, best_per = param_grid(x_train, x_test, y_train, y_test, device, cv_mccov, arch, iter_num, batch_size, dist_mode, d, func_mode)
            loss_m_val += best_per
            current_mccov_time = time.time() - start_time
            mccov_time += current_mccov_time
            print(f"MCCOV training end: {current_mccov_time:.2f}s")

    return loss_m_val, loss_w_val, mccov_time, wire_time
    
def main():
    func_modes = [0]  # 0 for III-1, 1 for III-2
    d_s = [1]
    x_modes = [-1]


    # Model information
    # bl = 0.01
    dist_mode = 1
    loss_modes = ['mccov','wire'] # Only MCCOV and WIRE are supported
    rep_num = 12
    arch = 'FNN_2'
    iter_num = 100
    batch_size = 64

    # --- GPU and parallel settings ---
    # List all your available GPUs
    available_devices = ['cuda:0', 'cuda:1', 'cuda:2', 'cuda:3', 'cuda:4', 'cuda:5'] # If you have multiple, e.g.: ['cuda:0', 'cuda:1', 'cuda:2']
    n_gpus = len(available_devices)
    # Number of parallel tasks should not exceed number of GPUs
    n_jobs = 2*n_gpus

    n_samples = [300, 500, 1000, 2000, 5000]  # , 2000, 5000
    p_dims = [100]

    cv_mccov = False
    cv_wire = False

    my_result = f"revise_result/sphere/"

    f = open(f"{my_result}test_p=100_rep=100", 'w')
    f.write(f"{my_result} start at {time.ctime(time.time())}\n")
    f.flush()

    nsdrfnormstimer = time.time()
    mccovtimer = 0
    wiretimer = 0

    for p_dim in p_dims:
        loss_m = np.zeros((rep_num, len(n_samples), len(func_modes), len(x_modes)))
        loss_w = np.zeros((rep_num, len(n_samples), len(func_modes), len(x_modes)))


        f.write(f"dim:{p_dim} ")

        for n, n_sample in enumerate(n_samples):
            for m, func_mode in enumerate(func_modes):
                d = d_s[m]
                for l, x_mode in enumerate(x_modes):
                    # Use joblib to run in parallel, assigning tasks to different GPUs in turn
                    results = Parallel(n_jobs=n_jobs)(delayed(run_repetition)(
                        n_sample, p_dim, func_mode, x_mode, loss_modes, dist_mode,
                        available_devices[rep % n_gpus], # Dynamically assign GPU
                        d, cv_wire, cv_mccov, arch, iter_num, batch_size
                    ) for rep in range(rep_num))

                    # Aggregate results
                    for rep, (loss_m_val, loss_w_val, mccov_time, wire_time) in enumerate(results):
                        loss_m[rep, n, m, l] = loss_m_val
                        loss_w[rep, n, m, l] = loss_w_val
                        mccovtimer += mccov_time
                        wiretimer += wire_time
                    
        for n, n_sample in enumerate(n_samples):
            if n == 0:
                f.write(f"{n_sample//1000}k ")
            else:
                f.write(f"       {n_sample//1000}k ")
            for m in range(len(func_modes)):
                f.write(f"func{m} ")
                for l in range(len(x_modes)):
                    for loss_mode in loss_modes:
                        if loss_mode=='mccov':
                            f.write(f" & {np.mean(loss_m[:, n, m, l]):.3f}({np.var(loss_m[:, n, m, l]):.3f})")
                        if loss_mode=='wire':
                            f.write(f" & {np.mean(loss_w[:, n, m, l]):.3f}({np.var(loss_w[:, n, m, l]):.3f})")
                    # f.write(f"c:{loss_c[n, m, l]:.3f} & m:{loss_m[n, m, l]:.3f} & g:{loss_g[n, m, l]:.3f} |")
                if m == len(func_modes)-1:
                    f.write("\n")
                else:
                    f.write("\n          ")
        f.flush()

    nsdrfnormstimer = (time.time() - nsdrfnormstimer)

    f.write(f"""MCCOV {(mccovtimer)/60:.3f} and WIRE {wiretimer/60:.3f} min \n""")

    # f.write(f"the {rep_num}th average distance correlation is {dcor_ave/rep_num:.3f} \n")
    # f.flush()

    f.write(f"""all complete at {time.ctime(time.time())}. NSDR consume total {(nsdrfnormstimer)/60:.3f}\n""")

    f.close()

if __name__ == "__main__":
    main()
