import argparse
import time, dcor
import numpy as np
import rpy2.robjects as robjects
from scipy.stats import multivariate_t, cauchy
from sklearn.model_selection import train_test_split
from NSDR import NSDR
from rpy2.robjects import r, numpy2ri
from rpy2.robjects.packages import importr
from support_function import *

from model_reg import *

# d=1 standard model

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

def data_generator_1(n, p, func_mode, x_mode):
    test_size = int(0.2*n)
    n = n + test_size  # add extra test samples on top of the original data
    x = x_generator(n, p, x_mode)
    y = func(x,func_mode)
    return train_test_split(x, y, test_size=test_size)

def data_generator_2(n, p, func_mode, x_mode, xt_mode):    # dedicated to outlier testing
    test_size =  int(0.2*n)                     # int(0.2*n)
    x = x_generator(n, p, x_mode)               # inject anomalous samples
    y = func(x,func_mode)
    x_t = x_generator(test_size, p, xt_mode)          # adjust the original distribution when needed
    y_t = func(x_t, func_mode)
    return (x,x_t,y,y_t)

def generate_gsir_res(x_train, x_test, y_train, y_test, d):
    importr("nsdr")
    rcode_snippet = '''
    # R code
    # You can access the NumPy array using the r_variable in R
    opt_candidate_x = 0.01  # 0.01
    opt_candidate_y = 0.01  # 0.01
    tryCatch({
    gsir_res <- gsir(x_train, x_test, y_train, 'scalar', opt_candidate_x, opt_candidate_y, 1, 1, r = d)
    res = gsir_res$suff.pred
    },error = function(err) {
        print(paste("MY ERROR: ", err))
    }
    )
    '''
    # new version
    from rpy2.robjects import conversion, default_converter
    with conversion.localconverter(default_converter + numpy2ri.converter):
        r.assign("x_train", x_train)
        r.assign("y_train", y_train)
        r.assign("x_test", x_test)
        r.assign("y_test", y_test)
        r.assign("d", d)
        r(rcode_snippet)
        gsir_res = np.array(robjects.r("res"))
        

    return gsir_res

def parse_args():
    parser = argparse.ArgumentParser(description="Run Euclidean simulation for structural dimension d=1.")
    parser.add_argument("--outlier", action="store_true", help="Enable the outlier setting.")
    parser.add_argument(
        "--case",
        choices=["one", "two"],
        default="one",
        help="Outlier case to use when --outlier is enabled.",
    )
    parser.add_argument("--result-name", default=None, help="Output file path for the saved results.")
    parser.add_argument(
        "--n-samples",
        nargs="+",
        type=int,
        default=None,
        help="Sample sizes to evaluate, e.g. --n-samples 500 800 1000.",
    )
    return parser.parse_args()


def main(args=None):
    if args is None:
        args = parse_args()

    # Test model robustness by using data with injected outliers
    outlier = args.outlier

    if not outlier:
        func_modes = [1]      # e.g., spheres in three-dimensional space
        x_modes = [0, 1, 3]      # x follows a standard normal distribution
    else:
        case = args.case    # one for X, two for epsilon
        if case == 'one':
            func_modes = [1]
            x_modes = [10]
            xt_mode = 1
        elif case == 'two':
            func_modes = [10]      # II_case3
            x_modes = [1]          # x follows a standard normal distribution
            xt_mode = 1           # II_case
    
    result_name = args.result_name or "revise_results_d1/test"
    rep_num = 100
    dist_mode = 0         # distance choice in the metric space: 0 Euclidean/Distribution, 1 Ball, 2 SPD
    loss_modes = ['mccov', 'gmdd', 'gsir', 'dyu']    # FCCov-Net, GMDD-Net, GSIR, DDR
    n_samples = args.n_samples or [500, 800, 1000]
    p_dims = [10, 20]
    categorical_y = False
    device = 'cuda:0'
    d = 1

    y_dcorfnorms = 0

    iter_num = 100
    f = open(f"{result_name}", 'w')
    f.write(f"{result_name} start at {time.ctime(time.time())}\n")

    batch_size = 64 # p*10, originally 500?
    f.flush()        # flush buffered data to the file immediately

    myclass = NSDR()

    # Timers
    nsdrfnormstimer = time.time()
    gmddtimer = 0
    mccovtimer = 0
    gsirtimer = 0
    dlintimer = 0
    dyutimer = 0

    for p_dim in p_dims:
        # Initialize tracking arrays
        loss_g = np.zeros((rep_num, len(n_samples), len(func_modes), len(x_modes)))
        loss_m = np.zeros((rep_num, len(n_samples), len(func_modes), len(x_modes)))
        loss_c = np.zeros((rep_num, len(n_samples), len(func_modes), len(x_modes)))
        loss_dl = np.zeros((rep_num, len(n_samples), len(func_modes), len(x_modes)))
        loss_dy = np.zeros((rep_num, len(n_samples), len(func_modes), len(x_modes)))
        # Initialize network structure
        net_fnorms = myclass.generate_default_net(p_dim, output_dim=d)
        net_fnorms.apply(myclass.init_xv_uniform)
        f.write(f"dim:{p_dim} ")

        for n, n_sample in enumerate(n_samples):
            for m, func_mode in enumerate(func_modes):
                for l, x_mode in enumerate(x_modes):
                    for rep in range(rep_num):
                        # Generate data
                        if not outlier:
                            x_train, x_test, y_train, y_test = data_generator_1(n=n_sample, p=p_dim, func_mode=func_mode, x_mode=x_mode)
                        else:
                            x_train, x_test, y_train, y_test = data_generator_2(n=n_sample, p=p_dim, func_mode=func_mode, x_mode=x_mode, xt_mode=xt_mode)
            

                        for loss_mode in loss_modes:
                            start_time = time.time()
                            if loss_mode == 'gsir':
                                start_time = time.time()
                                r_res = generate_gsir_res(x_train, x_test, y_train, y_test, d)
                                gsirtimer = gsirtimer + (time.time() - start_time)
                                r_dcor = dcor.distance_correlation(r_res, embedding(x_test, mode=func_mode))
                                loss_c[rep, n, m, l] += r_dcor
                            elif loss_mode=='dlin':
                                start_time = time.time()
                                dl_res = generate_mutualsdr_res(x_train, x_test, y_train, y_test, d)
                                dlintimer = dlintimer + (time.time() - start_time)
                                dl_dcor = dcor.distance_correlation(dl_res, embedding(x_test, mode=func_mode))
                                loss_dl[rep, n, m, l] += dl_dcor
                            elif loss_mode=='dyu':
                                start_time = time.time()
                                dy_res = generate_deepsdr_res(x_train, x_test, y_train, y_test, d)
                                dyutimer = dyutimer + (time.time() - start_time)
                                dy_dcor = dcor.distance_correlation(dy_res, embedding(x_test, mode=func_mode))
                                loss_dy[rep, n, m, l] += dy_dcor
                            elif loss_mode=='gmdd' or loss_mode=='mccov':  # 
                                if loss_mode == 'gmdd':
                                    bl = 1
                                else:
                                    bl=1e-4
                                model = NSDR(neural_network=net_fnorms, max_dim=1, method = "fnormsquare", adaptive_cv=False, retrain = True, 
                                            early_stop=False ,debug=True, device=device, initial_blambda = np.sqrt(d) * bl,dist_mode = dist_mode 
                                            ,loss_mode=loss_mode, iter_num=iter_num, categorical_y=categorical_y)
                                model.fit(x_train, y_train)
                                dr_train = model.transform(x_train)
                                dr_test = model.transform(x_test)
                            
                                y_dcorfnorms = dcor.distance_correlation(model.transform(x_test).astype(np.float64), 
                                                                            embedding(x_test, mode=func_mode).astype(np.float64)) 
                                if loss_mode=='gmdd':
                                    loss_g[rep, n, m, l] += y_dcorfnorms
                                    gmddtimer += time.time() - start_time
                                elif loss_mode=='mccov':
                                    loss_m[rep, n, m, l] += y_dcorfnorms
                                    mccovtimer += time.time() - start_time
                                elif loss_mode=='ccov':
                                    loss_c[rep, n, m, l] += y_dcorfnorms
                                del model
                        
        for n, n_sample in enumerate(n_samples):
            if n == 0:
                f.write(f"{n_sample/1000:.3f}k ")
            else:
                f.write(f"       {n_sample/1000:.3f}k ")
            for m in range(len(func_modes)):
                f.write(f"func{m} ")
                for l in range(len(x_modes)):
                    for loss_mode in loss_modes:
                        if loss_mode=='gsir':        # reuse the ccov output points
                            f.write(f" & {np.mean(loss_c[:, n, m, l]):.3f}({np.std(loss_c[:, n, m, l]):.3f}) ")
                        if loss_mode=='mccov':
                            f.write(f" & {np.mean(loss_m[:, n, m, l]):.3f}({np.std(loss_m[:, n, m, l]):.3f}) ")
                        if loss_mode=='gmdd':
                            f.write(f" & {np.mean(loss_g[:, n, m, l]):.3f}({np.std(loss_g[:, n, m, l]):.3f}) ")
                        if loss_mode=='dlin':
                            f.write(f" & {np.mean(loss_dl[:, n, m, l]):.3f}({np.std(loss_dl[:, n, m, l]):.3f}) ")
                        if loss_mode=='dyu':
                            f.write(f" & {np.mean(loss_dy[:, n, m, l]):.3f}({np.std(loss_dy[:, n, m, l]):.3f}) |\n")
                            f.write(" "*17)
                    # f.write(f"c:{loss_c[n, m, l]:.3f} & m:{loss_m[n, m, l]:.3f} & g:{loss_g[n, m, l]:.3f} |")
                if m == len(func_modes)-1:
                    f.write("\n")
                else:
                    f.write("\n          ")
        f.flush()

    nsdrfnormstimer = (time.time() - nsdrfnormstimer)

    f.write(f"""MCCOV {(mccovtimer)/60:.3f}, GMDD {(gmddtimer)/60:.3f}, GSIR {gsirtimer/60:.3f}, DLIN {dlintimer/60:.3f} and DYU {dyutimer/60:.3f} min \n""")

    # f.write(f"the {rep_num}th average distance correlation is {dcor_ave/rep_num:.3f} \n")
    # f.flush()

    f.write(f"""all complete at {time.ctime(time.time())}. NSDR consume total {(nsdrfnormstimer)/60:.3f}\n""")

    f.close()

if __name__ == "__main__":
    main()
