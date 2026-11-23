import copy, dcor, torch, scipy, os, time, _ctypes, warnings, matplotlib
import numpy as np
from numbers import Integral, Real
from torch import nn
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils import check_X_y, check_array, check_random_state
from sklearn.utils._param_validation import Interval, StrOptions
from sklearn.utils.validation import check_is_fitted
from sklearn.preprocessing import OneHotEncoder
from matplotlib import cm

from support_function import *



class NSDR(TransformerMixin, BaseEstimator):


    _parameter_constraints = {
    "initial_blambda": [Interval(Real, 0, None, closed="left"), StrOptions({'auto'})],
    "max_dim": [Interval(Integral, 0, None, closed="left")],
    "method": [StrOptions({"fnorm", "seq", "fnormsquare"})],
    "retrain": ["boolean"],
    "adaptive_cv": ["boolean"],
    "device": [StrOptions({'cpu', "cuda:0", "cuda:1"})],
    "debug": ["boolean"],
    "iter_num": [Interval(Integral, 1, None, closed="left")],
    "early_stop": ["boolean"],
    "y_mode": [Interval(Integral, 0, 5, closed="both")],
    "batch_size": [Interval(Integral, 12, None, closed="left")],
    "lr": [Interval(Real, 0, 1, closed="both")],
    "blambda_tilde": [Interval(Real, 0, None, closed="left"), StrOptions({"auto"})],
    "random_state": ["random_state"],
    "optimizer": [StrOptions({"adam", "sgd"})],
    "categorical_y": ["boolean"],
    "y_mode":[Interval(Integral, 0, 5, closed="both")],
    "loss_mode":[StrOptions({"gmdd", "mccov", 'ccov'})],
    "dist_mode":[Interval(Integral, 0, 4, closed="both")],
    }
    

    def __init__(self, neural_network=id(None), initial_blambda=1e-2, max_dim=1,
        method="seq", iter_num=100, early_stop=False, y_mode=0, dist_mode=0, random_state=None, 
        categorical_y=False, batch_size = 100, lr = 1e-3, blambda_tilde="auto", loss_mode='mccov', 
        adaptive_cv=False, retrain=True, optimizer="adam",device = 'cpu', 
        debug=False, debug_logfilename="default.txt"):
        self.set_params(neural_network = neural_network)
        self.initial_blambda = initial_blambda
        self.max_dim = max_dim
        self.retrain = retrain
        self.debug = debug
        self.adaptive_cv = adaptive_cv
        self.device = device
        self.iter_num = iter_num
        self.categorical_y = categorical_y
        self.early_stop = early_stop
        self.y_mode = y_mode
        self.dist_mode = dist_mode          
        self.loss_mode = loss_mode
        self.batch_size = batch_size
        self.optimizer = optimizer
        self.random_state = random_state
        self.lr = lr
        self.blambda_tilde = blambda_tilde
        self.method = method
        if isinstance(debug_logfilename, str):
            self.debug_logfilename = debug_logfilename
        else:
            raise ValueError("debug_logfilename should be a string specifying the directory")

    def set_params(self, **params):
        for key, value in params.items():
            if key == "neural_network":
                if isinstance(value, nn.Module):
                    self.neural_network = id(value)
                    self.neural_network_ = value
                elif isinstance(value, str):
                    try:
                        self.neural_network = id(eval(value))
                    except Exception as e:
                        raise ValueError(f"Evaluation fails for {value}")
                elif isinstance(value, int):
                    if value < int(hex(id(None))[:4] + "0" * (len(hex(id(None))) - 4), 16):
                        raise ValueError("u should not pass int as parameters")
                    self.neural_network = value
                else:
                    raise ValueError("neural_network should be specified correctly or leave it as default")
            else:
                super().set_params(**{key: value})
        return self

    def __getstate__(self):
        state = self.__dict__.copy()
        if self.debug:
            del state['file_']
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
    

    def _output_to_debugfile(self, word):
        if self.debug:
            print(word)
            self.file_.write(word)
            self.file_.write("\n")
            self.file_.flush()


    def _gmdd_dist(self, y, y_mode = 0):
        temp_y = y.reshape(y.shape[0], -1)
        tmp_dist_y = torch.cdist(temp_y, temp_y)
        if y_mode == 0:
            return tmp_dist_y
        elif y_mode == 1:
            return tmp_dist_y ** 2 / 2
        elif y_mode == 2:
            return tmp_dist_y ** 2 / (1 + tmp_dist_y ** 2)
        elif y_mode == 3:
            return 1 - torch.exp(- tmp_dist_y ** 2 / 2)
        elif y_mode == 4:
            return torch.log(1 + tmp_dist_y ** 2 / 2)
        elif y_mode == 5:
            return tmp_dist_y ** 1.5
        elif y_mode == 6:
            return (tmp_dist_y != 0).to(torch.float32)


    def load_pretrained_model(self, path_list):
        self.preloadnet_list_ = []
        if self.neural_network == id(None):
            raise Exception("u should specify the neural_network paramter before load the pretrained model")
        for path in path_list:
            net = copy.deepcopy(_ctypes.PyObj_FromPtr(self.neural_network))
            net = net.to(self.device)
            net.load_state_dict(torch.load(path))
            self.preloadnet_list_.append(net)


    def score(self, X, y):
        ypred = self.transform(X)
        return dcor.distance_correlation(ypred, y)


    def _gmdd_value(self, yhat, y):
        n_samples = y.shape[0]
        H = torch.eye(n_samples) - torch.ones(n_samples, 1) @ torch.ones(1, n_samples) / n_samples
        H = H.to(self.device)
        tmp_dist_y = self._gmdd_dist(y, y_mode=self.y_mode)
        tmp_yhat_mean = torch.mean(yhat, axis = 0).reshape(1, -1)
        tmp_centered_yhat = yhat - tmp_yhat_mean.repeat(n_samples, 1)
        A = H @ tmp_centered_yhat @ tmp_centered_yhat.T @ H
        B = H @ tmp_dist_y @ H
        gmdd = torch.mean(A * B)
        return gmdd










    def dist(self,y1,y2,mode):
        if mode==0:
            distances = torch.norm(y1-y2, dim=2)
        elif mode==1:
            distances = torch.arccos(torch.einsum('ijk,ijk->ij', y1, y2))
        elif mode==2:
            distances = []
            for i in range(y1.shape[0]):
                dis = spd_dis(y1[i],y2[i])
                distances.append(dis)
            distances = torch.stack(distances)
        elif mode==3:
            z1 = torch.sqrt(y1)
            z2 = torch.sqrt(y2)
            distances = torch.norm(z1-z2, dim=2)/np.sqrt(2)
        elif mode==4:
            distances = 0.5*torch.sum(torch.abs(y1-y2), dim=2)
        return distances    
    
    def calculate_distances_and_sort_torch(self, X, Y, mode=0):
        d = X.dim()
        n = X.shape[0]
        mean_Y = torch.mean(Y, dim=0) if Y.shape[1] > 1 else torch.mean(Y)
        Y_combined = torch.empty_like(Y)
        if d==2:
            expanded_X = X.unsqueeze(1).expand(-1, n, -1)
            distances = self.dist(expanded_X, expanded_X.transpose(0, 1), mode)
        elif d==3:
            expanded_X = X.unsqueeze(1).expand(-1, n, -1, -1)
            distances = self.dist(expanded_X, expanded_X.transpose(0, 1), mode)
    
        sorted_indices = torch.argsort(distances, dim=1)
        sorted_Y = torch.gather(Y.expand(n, -1, -1), 1, sorted_indices.unsqueeze(-1).expand(-1, -1, Y.shape[1]))
    
        Y_combined = sorted_Y - mean_Y
    
        return Y_combined
    
    
    def fast_cmp_torch(self, Y_sorted, Y):
        n = Y.shape[0]
        Y = Y - torch.mean(Y)
        Tnp = 0

        cumulative_sums = torch.cumsum(Y_sorted, dim=1)
        squared_cumsums = torch.sum(torch.square(cumulative_sums[:,:]), dim=1)
        Tnp += torch.sum(squared_cumsums)

        squared_Y_sorted = torch.square(Y_sorted)
        cumulative_sums_squares = torch.cumsum(squared_Y_sorted, dim=1)
        Tnp -= torch.sum(cumulative_sums_squares[:, :-1])

        Tnp -= 2 * torch.sum(Y @ cumulative_sums[:, :-1])

        Tnp += 2*(n-1)*torch.sum(Y**2)

        return Tnp / (n * (n-1) * (n-2) * (n-3))

    
    
    def mccov_value(self, yhat, y):
        mode = self.dist_mode
        yhat_sorted = self.calculate_distances_and_sort_torch(y, yhat, mode)
        num = 0.0

        if yhat.ndim == 1:
            num = self.fast_cmp_torch(yhat_sorted, yhat)
        elif yhat.ndim == 2:
            q = yhat.shape[1]
            for i in range(q):
                num += self.fast_cmp_torch(yhat_sorted[:, :, i], yhat[:, i])
        return -1.0*num

    def ccov_value(self, yhat, y):

        return -1.0*self.fast_ccov(yhat, y)
    
    
    
    def mysort_torch(self, X, Y):
        n = X.shape[0]
        sorted_Xs = []
    
        for i in range(Y.shape[1]):
            _, sorted_indices = Y[:, i].sort()
            sorted_X = X[sorted_indices]
        
            sorted_Xs.append(sorted_X)
        
        return torch.stack(sorted_Xs, dim=0)

    def fast_ccov(self, X, Y):
        X_mean = X - torch.mean(X)
        sorted_X = self.mysort_torch(X_mean, Y)
        n, p = Y.shape
        T_np = torch.tensor(0.0)
    
        cumsum_sorted_X = torch.cumsum(sorted_X, dim=1)
        cumsum_square_X = torch.cumsum(sorted_X**2, dim=1)
    
        T_np += torch.sum(torch.sum(cumsum_sorted_X[:,:n-1]**2, dim=1))*(n-2)*(n-3)
    
        J = torch.arange(1, n+1, dtype=torch.float32)
        coef_1 = 2 * (n*J - 2*n - 2*J + 2)
        T_np += torch.sum(coef_1[1:] * sorted_X[:,1:] * cumsum_sorted_X[:,:n-1])
    
        coef_2 = (n**2 - 2*n*J - n + 4*J - 4)
        T_np -= torch.sum(coef_2[1:] * cumsum_square_X[:,:n-1])
    
        T_np -= (n*(n**2-3*n+8)/3.) * torch.sum(cumsum_square_X[:,n-1])
    
        T_np += 2*torch.sum((torch.arange(0,n, dtype=torch.float32) * sorted_X)**2)
        
        return T_np/(n*(n-1)*(n-2)*(n-3)*(n-4))
    
    
    
    @staticmethod
    def generate_default_net(input_dim, first_hidden_width=None, last_hidden_width=16, increase_count=2,  output_dim=1):
        if first_hidden_width is None:
            first_hidden_width = 2 ** (int(np.log2(input_dim)) + 1)
        net = nn.Sequential()
        net.append(nn.Linear(input_dim, first_hidden_width))
        net.append(nn.ReLU())
        next_width = first_hidden_width
        for i in range(increase_count - 1):
            net.append(nn.Linear(next_width, next_width * 2))
            net.append(nn.ReLU())
            next_width = next_width * 2
        for i in range(increase_count - 1):
            net.append(nn.Linear(next_width, next_width // 2))
            net.append(nn.ReLU())
            next_width = next_width // 2
        if last_hidden_width is not None:
            while(next_width // 2 >= last_hidden_width):
                net.append(nn.Linear(next_width, next_width // 2))
                net.append(nn.ReLU())
                next_width = next_width // 2
        net.append(nn.Linear(next_width, output_dim))
        return net

    @staticmethod
    def init_xv_uniform(m):
        if type(m) == nn.Linear:
            nn.init.xavier_uniform_(m.weight)
            nn.init.zeros_(m.bias)

    def _generate_pred_inbatch(self, net, data_loader):
        net.eval()
        pred = torch.Tensor([]).to(self.device)
        with torch.no_grad():
            for data in data_loader:
                temp_x = data[0]
                yhat = net(temp_x)
                pred = torch.cat((pred, yhat), 0)
        return pred



    def _dist_cor(self, neural_network, data_loader):
        neural_network.eval()
        dc = 0
        with torch.no_grad():
            for temp_x, temp_y in data_loader:
                yhat = neural_network(temp_x)
                if temp_y.dim() <= 2:
                    dc += dcor.distance_correlation(yhat.detach().cpu().numpy(), temp_y.cpu().numpy())
                elif temp_y.dim() == 3:
                    ypred = embedding(temp_x)
                    dc += dcor.distance_correlation(yhat.detach().cpu().numpy(), ypred.cpu().numpy())
        dc = dc / len(data_loader)
        return dc



    def _gmdd_of_loader(self, neural_network, data_loader, standardize=True, batch_average=False):
        neural_network.eval()
        loader_gmdd = 0
        with torch.no_grad():
            if data_loader.batch_size < 10:
                raise Exception("batch size of data_loader in _gmdd_of_loader should be at least 10")
            if not batch_average and len(data_loader.dataset) > 5000:
                raise Exception("sample size of _gmdd_of_loader should be at most 5k if calculated without batch_average")
            if batch_average:
                with torch.no_grad():
                    for temp_x, temp_y in data_loader:
                        yhat = neural_network(temp_x)
                        if standardize:
                            loader_gmdd += self._gmdd_value(yhat, temp_y) / torch.sum(torch.diag(torch.cov(yhat.T).reshape(yhat.shape[1], -1)))
                        else:
                            loader_gmdd += self._gmdd_value(yhat, temp_y)
                loader_gmdd = loader_gmdd / len(data_loader)
            else:
                yhat = []
                y = []
                for temp_x, temp_y in data_loader:
                    y.append(temp_y)
                    yhat.append(neural_network(temp_x))

                y = torch.cat(y, dim=0)
                yhat = torch.cat(yhat, dim=0)
                if standardize:
                    loader_gmdd += self._gmdd_value(yhat, y) / torch.sum(torch.diag(torch.cov(yhat.T).reshape(yhat.shape[1], -1)))
                else:
                    loader_gmdd += self._gmdd_value(yhat, y)

        return loader_gmdd

    def get_sample_gmdd_value(self, X, y, standardize=True, batch_average=None):
        check_is_fitted(self)
        if batch_average is None:
            if X.shape[0] <= 5000:
                batch_average = False
            else:
                batch_average = True
        X = torch.Tensor(X).to(self.device)
        y = torch.Tensor(y).to(self.device)
        if self.categorical_y and (y.ndim == 1 or y.shape[1] == 1):
            y = torch.nn.functional.one_hot(y.to(torch.long))
            y = y.to(torch.float32).reshape(y.shape[0], -1)
        gmdd = 0
        data_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(X, y), batch_size=self.batch_size, shuffle=True)
        if self.method == "seq":
            for i in range(self.max_dim):
                gmdd += self._gmdd_of_loader(self.estimators_[i], data_loader, standardize, batch_average)
        else:
            gmdd = self._gmdd_of_loader(self.estimators_[0], data_loader, standardize, batch_average)
        return gmdd.data.cpu().numpy()

    def get_sample_gmdd_matrix(self, X, y):
        check_is_fitted(self)
        y = torch.Tensor(y).to(self.device)
        if self.categorical_y and (y.ndim == 1 or y.shape[1] == 1):
            y = torch.nn.functional.one_hot(y.to(torch.long))
            y = y.to(torch.float32).reshape(y.shape[0], -1)

        n_samples = y.shape[0]
        yhat = torch.Tensor(self.transform(X)).to(self.device)
        if n_samples > 5000:
            warnings.warn("It's not preferred to calculate the empirical gmdd matrix with large sample size")
        tmp_yhat_mean = torch.mean(yhat, axis = 0).reshape(1, -1)
        tmp_centered_yhat = yhat - tmp_yhat_mean.repeat(n_samples, 1)
        gmddsize = yhat.shape[1]
        gmdd = torch.zeros((gmddsize, gmddsize))
        cdisty = self._gmdd_dist(y, y_mode=self.y_mode)
        for i in range(gmddsize):
            for j in range(i, gmddsize):
                gmdd[i, j] = torch.mean(torch.outer(tmp_centered_yhat[:, i], tmp_centered_yhat[:, j]) * cdisty)
        tindx = torch.tril_indices(gmddsize, gmddsize, -1)
        gmdd[tindx[0], tindx[1]] = gmdd.T[tindx[0], tindx[1]]
        return gmdd.data.cpu().numpy()

    @staticmethod
    def torch_normsq(x):
        return (torch.norm(x)) ** 2

    def fit(self, X, y):
        torch.set_default_dtype(torch.float32)
        self._validate_params()
        self._rng = check_random_state(self.random_state)
        torch.manual_seed(self._rng.randint(0, 0xffff_ffff))


        if self.categorical_y:
            self.y_mode = 6

        if self.blambda_tilde == "auto":
            self.blambda_tilde_ = 1
        else:
            self.blambda_tilde_  = self.blambda_tilde

        if self.neural_network == id(None):
            if self.method == "seq":
                self.nn_output_dim_ = 1
            else:
                self.nn_output_dim_ = y.shape[1]
            self.neural_network_ = self.generate_default_net(X.shape[1], 2 ** (int(np.log2(X.shape[1]) +1)), 16, 2, self.nn_output_dim_)
        else:
            self.neural_network_ = _ctypes.PyObj_FromPtr(self.neural_network)
        keyrev = reversed(self.neural_network_.state_dict())
        self.nn_output_dim_ = self.neural_network_.state_dict()[next(keyrev)].shape[0]

        if self.initial_blambda == "auto":
            if self.nn_output_dim_ == 1:
                self.initial_blambda_ = 0.1 ** 6
            else:
                self.initial_blambda_ = 1
        else:
            self.initial_blambda_ = self.initial_blambda

        if self.method == "seq" and self.nn_output_dim_ > 1:
            raise ValueError("Output dimension of neural network is greater than 1, u should use fnorm method instead of sequential method")
        if self.max_dim > 1 and self.method != "seq":
            raise ValueError("Sequential method is preferred u train NSDR with max_dim greater than 1")
        if self.debug:
            if (not os.path.isfile(self.debug_logfilename)) and (not os.path.exists(os.path.dirname(self.debug_logfilename))):
                raise Exception("debug logfile not exist")
            self.file_ = open(self.debug_logfilename, "w")
            self.file_.write(f"NSDR start at {time.ctime(time.time())} \n")
            self.file_.flush()

        if self.method == "fnorm":
            self.normfunc_ = torch.norm
            self.lower_cov_ = 0
            self.upper_cov_ = np.inf
        elif self.method == "seq":
            self.normfunc_ = self.torch_normsq
            self.lower_cov_ = 0
            self.upper_cov_ = np.inf
        elif self.method == "fnormsquare":
            self.normfunc_ = self.torch_normsq
            self.lower_cov_ = 0
            self.upper_cov_ = np.inf
        else:
            raise Exception("norm should be fnorm or seq")


        if self.retrain:
            if not hasattr(self, 'preloadnet_list_'):
                self.preloadnet_list_ = []
        else:
            if not hasattr(self, 'preloadnet_list_'):
                raise Exception("u should preload paramter of net list before fitting")
        self.estimators_ = []
        X = torch.Tensor(X).to(self.device)
        y = torch.Tensor(y).to(self.device)
        if self.categorical_y and (y.ndim == 1 or y.shape[1] == 1):
            y = torch.nn.functional.one_hot(y.to(torch.long))
            y = y.to(torch.float32).reshape(y.shape[0], -1)


        self._fit_stages(X, y)

        return self




    def _fit_stages(self, X, y):
        self.estimators_.extend(self.preloadnet_list_)
        shuffled_dataset = torch.utils.data.Subset(torch.utils.data.TensorDataset(X, y), torch.randperm(len(X)).tolist())
        train_loader = torch.utils.data.DataLoader(shuffled_dataset, batch_size=self.batch_size, shuffle=False)
        for i in range(len(self.preloadnet_list_), self.max_dim):
            self.estimators_.append(copy.deepcopy(self.neural_network_))

        for i in range(self.max_dim):
            self.estimators_[i] = self.estimators_[i].to(self.device)

        for i in range(self.max_dim):
            if i >= len(self.preloadnet_list_) or (i < len(self.preloadnet_list_) and self.retrain):
                fitted = self._fit_stage(i, X, y)
            else:
                fitted = self.estimators_[i]
            fitted.eval()
            pred = self._generate_pred_inbatch(fitted, train_loader)
            if self.method == "seq":
                keyrev = reversed(fitted.state_dict())
                fitted_lastlayer_bias = fitted.state_dict()[next(keyrev)]
                fitted_lastlayer_weight = fitted.state_dict()[next(keyrev)]
                fitted_lastlayer_weight.copy_(fitted_lastlayer_weight/torch.sqrt(torch.cov(pred.T)))
                pred = self._generate_pred_inbatch(fitted, train_loader)
                fitted_lastlayer_bias.copy_(fitted_lastlayer_bias - torch.mean(pred, axis=0))

            if self.method == "seq" and i == 0 and self.blambda_tilde == "auto":
                if len(train_loader.dataset) <= 5000:
                    batch_average = False
                else:
                    batch_average = True
                self.blambda_tilde_ = (-self._gmdd_of_loader(fitted, train_loader, standardize=True, batch_average=batch_average)).detach()
                self._output_to_debugfile(f"self.blambda_tilde_ {self.blambda_tilde_}")

            self.estimators_[i] = fitted
        pred = self._generate_pred_inbatch(fitted, train_loader)
        if pred.shape[1] > 1 and self.debug:
            self._output_to_debugfile(f"""corelation of net is {torch.corrcoef(pred.T)} and 
                the fnorm of corelation is {torch.norm(torch.corrcoef(pred.T) - torch.eye(pred.shape[1], device=self.device)) / pred.shape[1]} \n""")
                

    def _fit_stage(self, i, X, y):
        neural_network = self.estimators_[i]
        training_dataset = torch.utils.data.TensorDataset(X, y)
        train_loader = torch.utils.data.DataLoader(training_dataset, batch_size=self.batch_size, shuffle=True)
        tr_dataset, cv_dataset = torch.utils.data.random_split(training_dataset, [0.5, 0.5])
        tr_loader = torch.utils.data.DataLoader(tr_dataset, batch_size=self.batch_size, shuffle=True)
        cv_loader = torch.utils.data.DataLoader(cv_dataset, batch_size=self.batch_size, shuffle=True)
        precision = 6
        upper_blambda = 10 ** precision
        current_blambda = self.initial_blambda_
        lower_blambda = 0.1 ** precision
        iterated = set([current_blambda])
        old_discor = 0
        dec_count = 0
        best_blambda = current_blambda
        best_dis = 0
        while self.adaptive_cv:
            self.blambda_ = current_blambda
            fitted_nn = self._fit_one_neural_network(i, tr_loader)
            new_discor = self._dist_cor(fitted_nn, cv_loader)
            self._output_to_debugfile(f"blambda is {current_blambda} discor is {new_discor} cov sum is {self.cov_sum_[-1]}")
            if new_discor > best_dis and self.lower_cov_ <= self.cov_sum_[-1] <= self.upper_cov_:
                best_dis = new_discor
                best_blambda = current_blambda
                self._output_to_debugfile(f"best discor is {best_dis} best blambda is {best_blambda}")
            if self.cov_sum_[-1] < 10 ** 2:
                upper_blambda = current_blambda
                current_blambda = current_blambda / 10
                current_blambda = np.round(current_blambda, precision)
                if current_blambda < lower_blambda or current_blambda in iterated:
                    current_blambda = 10 ** ((np.log10(upper_blambda * lower_blambda)) / 2)
                current_blambda = np.round(current_blambda, precision)
            else:
                lower_blambda = current_blambda
                current_blambda = current_blambda * 10
                current_blambda = np.round(current_blambda, precision)
                if current_blambda > upper_blambda or current_blambda in iterated:
                    current_blambda = 10 ** ((np.log10(upper_blambda * lower_blambda)) / 2)
                current_blambda = np.round(current_blambda, precision)
            iterated.add(current_blambda)
            if (upper_blambda - lower_blambda) / 10 ** (np.floor(np.log10(lower_blambda))) < 3:
                self._output_to_debugfile(f"last cross validate with blambda within interval ({lower_blambda}, {upper_blambda})")
                minor_num = 5
                step = (upper_blambda - lower_blambda) / (minor_num + 2)
                minor_set = np.linspace(lower_blambda + step, upper_blambda - step, minor_num)
                for _ in range(minor_num):
                    current_blambda = minor_set[_]
                    self.blambda_ = current_blambda
                    fitted_nn = self._fit_one_neural_network(i, tr_loader)
                    new_discor = self._dist_cor(fitted_nn, cv_loader)
                    if new_discor > best_dis and self.lower_cov_ <= self.cov_sum_[-1] <= self.upper_cov_:
                        best_dis = new_discor
                        best_blambda = current_blambda
                        self._output_to_debugfile(f"best discor is {best_dis} best blambda is {best_blambda} at last cv cov sum is {self.cov_sum_[-1]}")
                break
        if self.adaptive_cv:
            self._output_to_debugfile(f"blambda adaptive cv has done with best blambda {best_blambda} and best discor {best_dis}")
        current_blambda = best_blambda
        refit_num = 0
        while True:
            self.blambda_ = current_blambda
            if self.early_stop:
                fitted_nn = self._fit_one_neural_network(i, tr_loader, cv_loader)
            else:
                fitted_nn = self._fit_one_neural_network(i, train_loader)

            if self.max_dim == 1 and self.nn_output_dim_ == 1:
                self._output_to_debugfile("no validation for nn with output_dim 1 and max dim 1")
                break

            if self.lower_cov_ <= self.cov_sum_[-1] <= self.upper_cov_:
                if self.method != "seq":
                    pred = self._generate_pred_inbatch(fitted_nn, train_loader)
                    if pred.shape[1] == 1:
                        break
                    normdiff = torch.norm(torch.corrcoef(pred.T) - torch.eye(pred.shape[1], device=self.device)) / pred.shape[1]
                    if normdiff >= 0.8:
                        self._output_to_debugfile(f"fnorm correlated, {torch.corrcoef(pred.T)[:2, :2]}, diff {normdiff} ,blambda {self.blambda_}")
                        best_blambda = best_blambda * 1.1
                        current_blambda = best_blambda
                    else:
                        self._output_to_debugfile(f"network traininng done with blambda {best_blambda:.2f}, normdiff {normdiff} total refit time is {refit_num} cov sum is {self.cov_sum_[-1]}")
                        break
                elif i >= 1:
                    pred = self._generate_pred_inbatch(fitted_nn, train_loader)
                    for j in range(i):
                        nn = self.estimators_[j]
                        pred = torch.cat([pred, self._generate_pred_inbatch(nn, train_loader)], axis=1)
                    normdiff = torch.norm(torch.corrcoef(pred.T) - torch.eye(pred.shape[1], device=self.device)) / pred.shape[1]
                    self._output_to_debugfile(f"cov {self.cov_sum_[-1]} seq cor nom {normdiff}")
                    if normdiff >= 0.8:
                        self._output_to_debugfile(f"seq correlated, {torch.corrcoef(pred.T)}, cov {self.cov_sum_[-1]}, tilde {self.blambda_tilde_}, lamb {current_blambda}")
                        self.blambda_tilde_ = self.blambda_tilde_ * 1.1
                    else:
                        self._output_to_debugfile(f"done, {torch.corrcoef(pred.T)}, cov {self.cov_sum_[-1]}, tilde {self.blambda_tilde_}, lamb {current_blambda}")
                        self._output_to_debugfile(f"network traininng done with blambda {best_blambda} total refit time is {refit_num} cov sum is {self.cov_sum_[-1]}")
                        break
                else:
                    self._output_to_debugfile(f"network traininng done, blambda_tilde {self.blambda_tilde_},blambda {best_blambda} total refit time is {refit_num} cov sum is {self.cov_sum_[-1]}")
                    break
            else:
                refit_num += 1
                if refit_num % 2 == 0:
                    self._output_to_debugfile(f"{refit_num} refit, reinital with covsum {self.cov_sum_[-1]}")
                if refit_num % 2 == 0:
                    if self.method == "seq":
                        print(f"cov small {self.cov_sum_[-1]}, tilde {self.blambda_tilde_}, lamb {current_blambda}")
                    else:
                        if self.cov_sum_[-1] < self.lower_cov_:
                            best_blambda = best_blambda / 1.1
                        else:
                            best_blambda = best_blambda * np.log(self.cov_sum_[-1]) / np.log(self.upper_cov_)
                    current_blambda = best_blambda
                if refit_num % 21 == 0:
                    print(f"cov small {self.cov_sum_[-1]}, tilde {self.blambda_tilde_}, lamb {current_blambda}")
                    break
        y_mode_emb_distcor = self._dist_cor(fitted_nn, train_loader)
        if np.isnan(y_mode_emb_distcor) or y_mode_emb_distcor == 0:
            print(y)
            print("y may be the same for all sample")
        return fitted_nn



    def _fit_one_neural_network(self, i, tr_loader, cv_loader=None):
        self.gmdd_loss_ = []
        self.total_loss_ = []
        self.cov_loss_ = []
        self.cov_sum_ = []
        self.dc_hist_ = []
        neural_network = copy.deepcopy(self.estimators_[i])
        if self.optimizer == "adam":
            optimizer = torch.optim.Adam(neural_network.parameters(), lr = self.lr)
        elif self.optimizer == "sgd":
            optimizer = torch.optim.SGD(neural_network.parameters(), lr= self.lr, momentum=0.9)
        best_dc = 0
        best_iternum = 0
        consecutive_decrease_count = 0
        epoch_per_test = 10
        for i_iter in range(self.iter_num):
            self._fit_one_epoch(neural_network, optimizer, i, tr_loader)
            if i_iter % epoch_per_test == 0 and self.early_stop:
                dc = self._dist_cor(neural_network, cv_loader)
                self.dc_hist_.append(dc)
                if dc > best_dc:
                    best_dc = dc
                    best_iternum = i_iter
                    consecutive_decrease_count = 0
                    self._output_to_debugfile(f"{i_iter}th iteration with best cv distance_correlation {dc}")
                else:
                    consecutive_decrease_count += 1

            if consecutive_decrease_count >= int(100/epoch_per_test):
                self._output_to_debugfile(f"No distance_correlation improvement on cv loader for consecutive iter at {best_iternum}")
                break

        if cv_loader is not None:
            for i_iter in range(best_iternum):
                self._fit_one_epoch(neural_network, optimizer, i, cv_loader)
                dc = self._dist_cor(neural_network, cv_loader)
                self.dc_hist_.append(dc)
            self._output_to_debugfile(f"early_stop stop at iter {i_iter + 1} with best_iternum {best_iternum}")

        return neural_network

    def _fit_one_epoch(self, neural_network, optimizer, i, tr_loader):
        if self.loss_mode=='mccov':
            loss_func = self.mccov_value
        elif self.loss_mode=='gmdd':
            loss_func = self._gmdd_value
        elif self.loss_mode=='ccov':
            loss_func = self.ccov_value
        else:
            print("error")
        for temp_x, temp_y in tr_loader:
            yhat = neural_network(temp_x)
            yhatdim = yhat.shape[1]
            tmp_yhat_mean = torch.mean(yhat, axis = 0).reshape(1, -1)
            tmp_centered_yhat = yhat - tmp_yhat_mean.repeat(yhat.shape[0], 1)
            gmdd = loss_func(yhat, temp_y)
            covsum = torch.sum(torch.abs(torch.cov(tmp_centered_yhat.T)))
            cov_loss = self.blambda_ * self.normfunc_(torch.cov(tmp_centered_yhat.T).reshape(yhatdim, yhatdim) - torch.eye(yhatdim).to(self.device))
            culmulated = 0
            for j in range(i):
                net_ = self.estimators_[j]
                previous = net_(temp_x).detach()
                previous = (previous - torch.mean(previous, axis=0).reshape(1, -1).repeat(temp_x.shape[0], 1)).detach()
                culmulated += (previous.T.detach() @ tmp_centered_yhat / temp_x.shape[0]) ** 2

            cov_loss = cov_loss + self.blambda_tilde_ * culmulated
            loss = cov_loss + gmdd


            self.total_loss_.append(loss.data.cpu().tolist())
            self.gmdd_loss_.append(gmdd.data.cpu().tolist())
            self.cov_loss_.append(cov_loss.data.cpu().tolist())
            self.cov_sum_.append(covsum.data.cpu().tolist())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    
    def transform(self, X):
        check_is_fitted(self)
        X = self._validate_data(X, allow_nd=True, reset=False)
        data_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(torch.Tensor(X).to(self.device), ), batch_size=100, shuffle=False)

        if self.method == "seq":
            z = torch.zeros((X.shape[0], self.max_dim)).to(self.device)
            for i in range(self.max_dim):
                self.estimators_[i].eval()
                z[:, i:(i+1)] = self._generate_pred_inbatch(self.estimators_[i], data_loader)
        else:
            z = self._generate_pred_inbatch(self.estimators_[0], data_loader)
        z = np.array(z.detach().cpu().data, dtype = np.float32)
        
        return z

    def predict(self, X):
        return self.transform(X)
