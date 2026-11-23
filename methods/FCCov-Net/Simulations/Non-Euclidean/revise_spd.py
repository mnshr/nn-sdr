import os
import torch, time, dcor, os
import numpy as np
from scipy.stats import multivariate_t
from support_function import *
from mynet import *
from joblib import Parallel, delayed

# SPD data generation
def func(x, mode):
    f_1 = 0.5*x[:,0]+0.5*x[:,1]
    f_2 = 0.5*x[:,0]+0.5*x[:,1]
    sigma = 0.2
    if mode==0:     
        rho_1 = np.sin(4*np.pi*f_1*(2*f_2-1))
        z = expm(d2_Nmm(rho_1, sigma))

    elif mode==1:   
        rho_1 = x[:, 0] / (1 + np.abs(x[:, 1])**0.5)
        rho_2 = np.sin(x[:, 2]**2) + np.exp(x[:, 3]**2)
        off_diag = np.column_stack((rho_1, rho_2))
        z = expm(d3_Nmm(off_diag, sigma))
    
    return z

def func_1(x, mode):
    n = x.shape[0]
    f_1 = 0.5*x[:,0]+0.5*x[:,1]
    f_2 = 0.5*x[:,0]+0.5*x[:,1]
    if mode==0:             
        z = np.zeros((n,1))
        z[:,0] = np.sin(4*np.pi*f_1*(2*f_2-1))

    elif mode==1:           
        z = np.zeros((n,2))
        z[:,0] = x[:, 0] / (1 + np.abs(x[:, 1])**0.5)
        z[:,1] = np.sin(x[:, 2]**2) + np.exp(x[:, 3]**2)
        
    return z

embedding = func_1

def x_generator(n, p, mode):
    mean = np.zeros(p)
    cov = np.zeros((p, p))
    for i in range(p):
        for j in range(p):
            cov[i, j] = 0.5 ** abs(i - j)

    if mode==-1:         
        x = np.random.uniform(0, 1, (n,p))
    if mode==0:          
        x = np.random.normal(0, 1, (n,p))
    elif mode==1:      
        x = np.random.multivariate_normal(mean, cov, n)
    elif mode==2:       
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

# Distance matrix for WIRE
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
    if mode==6:  # SPD log Cholesky distance
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

def param_grid(x_train, x_test, y_train, y_test, device, cv_mccov, arch, iter_num, batch_size, dist_mode, d, func_mode, init_weights_path=None, success_threshold=0.35):
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
        
        # Return the tuple with the best y_dcorfnorms
        best_bl, best_per = max(performance, key=lambda x: x[1])
        print("best parameters:",best_bl)
        return (best_bl, best_per)
    else:
        bl=1e-5
        model = CustomNet(architecture=arch, optimizer_name='Adam', iter_num=iter_num, p_dim=p, batch_size=batch_size,
                          blambda=bl, dist_mode=dist_mode, device=device, output_dim=d)
        
        # Before training, try to load successful weights
        if init_weights_path and os.path.exists(init_weights_path):
            try:
                print(f"Found successful weights at {init_weights_path}. Loading them.")
                model.load_state_dict(torch.load(init_weights_path, map_location=device))
            except Exception as e:
                print(f"Could not load weights: {e}")

        model.train_step(x_train, y_train, model.mccov_loss)
        y_hat = model.transform(x_test).astype(np.float64)
        y_dcorfnorms = dcor.distance_correlation(y_hat, embedding(x_test, mode=func_mode))

        # After training, if successful, save weights for other processes to use
        # Use "not os.path.exists" to avoid file write race conditions; the first successful process will create the file
        if init_weights_path and y_dcorfnorms > success_threshold and not os.path.exists(init_weights_path):
            print(f"Run succeeded with dcor={y_dcorfnorms:.4f}. Saving weights to {init_weights_path}.")
            torch.save(model.state_dict(), init_weights_path)
        elif init_weights_path and not os.path.exists(init_weights_path):
            print(f"Run failed with dcor={y_dcorfnorms:.4f}. Not saving weights.")

        return (bl, y_dcorfnorms)
    
def run_repetition(n_sample, p_dim, func_mode, x_mode, loss_modes, dist_mode, device, d, cv_wire, cv_mccov, arch, iter_num, batch_size, init_weights_path, success_threshold=0.35):
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
            D = distance_matrix(y_train, dist_mode, 'cpu') # Compute distance matrix, WIRE method uses CPU by default
            # Cross-validation for WIRE parameters
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
            best_bl, best_per = param_grid(x_train, x_test, y_train, y_test, device, cv_mccov, arch, iter_num, batch_size, dist_mode, d, func_mode, init_weights_path=init_weights_path, success_threshold=success_threshold)
            loss_m_val += best_per
            current_mccov_time = time.time() - start_time
            mccov_time += current_mccov_time
            print(f"MCCOV training end: {current_mccov_time:.2f}s")

    return loss_m_val, loss_w_val, mccov_time, wire_time

def pre_search_for_good_initialization(n_sample, p_dim, func_mode, x_mode, loss_modes, dist_mode, device, d, cv_wire, cv_mccov, arch, iter_num, batch_size, init_weights_path, success_threshold=0.35):
    """
    Continuously run single training instances until a successful set of weights is found and saved.
    """
    print("--- Starting pre-search for a good initialization --- ")
    # Clean up old weight files to ensure search starts from scratch
    if os.path.exists(init_weights_path):
        os.remove(init_weights_path)

    run_count = 0
    while not os.path.exists(init_weights_path):
        run_count += 1
        print(f"Pre-search attempt #{run_count}...")
        # 运行一次训练
        run_repetition(
            n_sample, p_dim, func_mode, x_mode, loss_modes, dist_mode, 
            device, d, cv_wire, cv_mccov, arch, iter_num, batch_size, 
            init_weights_path,
            success_threshold
        )
    
    print(f"--- Successful initialization found and saved to {init_weights_path} ---")

def main():
    scenarios = {
        0: 1,  # func_mode=0, d=1
        1: 2,  # func_mode=1, d=2
    }
    x_modes = [-1]

    my_result = f"revise_result/spd/test"  #

    # parameters to control pre-search ---
    force_pre_search = False  # Set to True to force re-search, set to False to load directly if file exists
    success_threshold = 0.35  # Define threshold for success

    # --- GPU and parallel settings ---
    # List all available GPUs
    available_devices = ['cuda:0', 'cuda:1', 'cuda:2', 'cuda:3']
    n_gpus = len(available_devices)
    # Number of parallel tasks should not exceed number of GPUs
    n_jobs = 2*n_gpus 

    # Model information
    dist_modes = [2, 6] # 0 for Euclidean, 1 for spherical, 2 for SPD affine distance, 3 for Hellinger, 4 for TV, 5 for categorical, 6 for SPD log-cholesky distance
    loss_modes = ['mccov', 'wire'] # Only these two: 'mccov', 'wire'
    rep_num = 100
    arch = 'FNN_2'
    iter_num = 100
    batch_size = 64     # If sample size is 50, setting to 32 is more appropriate

    n_samples = [50, 500, 1000, 2000, 5000]  
    p_dims = [100]

    cv_mccov = False
    cv_wire = False

    f = open(f"{my_result}", 'w')
    f.write(f"{my_result} start at {time.ctime(time.time())}\n\n")
    f.flush()

    nsdrfnormstimer = time.time()
    mccovtimer = 0
    wiretimer = 0

    for p_dim in p_dims:
        f.write(f"dim:{p_dim}\n")
        for dist_mode in dist_modes:
            f.write(f"dist_mode:{dist_mode}\n")
            loss_m = np.zeros((rep_num, len(n_samples), len(scenarios), len(x_modes)))
            loss_w = np.zeros((rep_num, len(n_samples), len(scenarios), len(x_modes)))
            
            for n, n_sample in enumerate(n_samples):
                for m, (func_mode, d) in enumerate(scenarios.items()):
                    for l, x_mode in enumerate(x_modes):
                        # Define weight path for each configuration
                        init_weights_path = os.path.join(os.path.dirname(my_result), f"spd_init_dist={dist_mode}_func={func_mode}_n={n_sample}.pt")

                        # Decide whether to perform pre-search based on parameters and file existence (only when loss_modes contains mccov)
                        if 'mccov' in loss_modes:
                            if force_pre_search or not os.path.exists(init_weights_path):
                                print(f"Pre-search needed for dist={dist_mode}, func={func_mode}, n={n_sample}. force_pre_search={force_pre_search}, file_exists={os.path.exists(init_weights_path)}")
                                pre_search_for_good_initialization(
                                    n_sample, p_dim, func_mode, x_mode, ['mccov'], dist_mode, # Only use mccov for pre-search
                                    available_devices[0], # Use the first GPU for search
                                    d, cv_wire, cv_mccov, arch, iter_num, batch_size,
                                    init_weights_path,
                                    success_threshold=success_threshold
                                )
                            else:
                                print(f"Skipping pre-search. Found existing weights at {init_weights_path}")
                        else:
                            init_weights_path = None  # When not using mccov, pretrained weights are not needed

                        # Use joblib to run in parallel and assign tasks to different GPUs in rotation
                        print(f"--- Starting main parallel computation for dist={dist_mode}, func={func_mode}, n={n_sample} ---")
                        results = Parallel(n_jobs=n_jobs)(delayed(run_repetition)(
                            n_sample, p_dim, func_mode, x_mode, loss_modes, dist_mode, 
                            available_devices[rep % n_gpus], # Dynamically assign GPUs
                            d, cv_wire, cv_mccov, arch, iter_num, batch_size,
                            init_weights_path  # [New feature] Pass path
                        ) for rep in range(rep_num))
                        
                        # Aggregate results
                        for rep, (loss_m_val, loss_w_val, mccov_time, wire_time) in enumerate(results):
                            loss_m[rep, n, m, l] = loss_m_val
                            loss_w[rep, n, m, l] = loss_w_val
                            mccovtimer += mccov_time
                            wiretimer += wire_time
            
            # Write results
            f.write("-----------\n")

            single_val_width = 14  # "0.500 (0.012)"
            separator = " & "
            d_col_width = len(loss_modes) * single_val_width + (len(loss_modes) - 1) * len(separator) + 2

            # First row: d value headers
            header = " " * 10
            for func_mode, d in scenarios.items():
                header += " | " + f"d = {d}".ljust(d_col_width)
            f.write(header + "\n")

            # Second row: method name headers
            method_header = " " * 10
            for func_mode, d in scenarios.items():
                methods_str = separator.join(loss_modes)
                method_header += " | " + methods_str.ljust(d_col_width)
            f.write(method_header + "\n")

            # Data rows
            for n, n_sample in enumerate(n_samples):
                line = f"{n_sample:<10}"
                for m, func_mode in enumerate(scenarios.keys()):
                    values = []
                    for l, x_mode in enumerate(x_modes):
                        for loss_mode in loss_modes:
                            if loss_mode == 'mccov':
                                mean_val = np.mean(loss_m[:, n, m, l])
                                var_val = np.var(loss_m[:, n, m, l])
                                values.append(f"{mean_val:.3f} ({var_val:.3f})")
                            elif loss_mode == 'wire':
                                mean_val = np.mean(loss_w[:, n, m, l])
                                var_val = np.var(loss_w[:, n, m, l])
                                values.append(f"{mean_val:.3f} ({var_val:.3f})")
                    line += " | " + separator.join(values).ljust(d_col_width)
                f.write(line + "\n")
            f.flush()

    nsdrfnormstimer = (time.time() - nsdrfnormstimer)

    f.write("\n" + "="*40 + "\n")
    f.write("           SUMMARY\n")
    f.write("="*40 + "\n")
    f.write(f"Total MCCOV time: {mccovtimer / 60:.3f} min\n")
    f.write(f"Total WIRE time:     {wiretimer / 60:.3f} min\n")
    f.write(f"Total execution time: {nsdrfnormstimer / 60:.3f} min\n")
    f.write(f"All complete at {time.ctime(time.time())}\n")
    f.write("="*40 + "\n")

    f.close()                                

if __name__ == "__main__":
    main()