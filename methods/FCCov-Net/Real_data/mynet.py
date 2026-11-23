import torch
import torch.nn as nn
import torch.nn.init as init
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, TensorDataset
import dcor
import time
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from torchvision.models import resnet18
from torchvision import models
from support_function import *

# Use Kaiming (He) initialization for fully connected layers, a good choice for ReLU activation
def init_weights(m):
    if isinstance(m, nn.Linear):
        init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        if m.bias is not None:
            init.constant_(m.bias, 0)           

class ResNet18(nn.Module):
    def __init__(self, num_classes=2):
        super(ResNet18, self).__init__()
        self.model = resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        self.model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)  # Modify input layer
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)  # Modify output layer
    
    def forward(self, x):
        return self.model(x)   

class CustomDenseNet121(nn.Module):
    def __init__(self, output_dim):
        super(CustomDenseNet121, self).__init__()
        # Load DenseNet-121 model
        self.densenet = models.densenet121(weights=None)

        # Modify first convolutional layer to accept single-channel grayscale images
        self.densenet.features.conv0 = nn.Conv2d(
            1, 64, kernel_size=7, stride=2, padding=3, bias=False)

        # Modify final classification layer
        num_ftrs = self.densenet.classifier.in_features
        self.densenet.classifier = nn.Linear(num_ftrs, output_dim)

    def forward(self, x):
        return self.densenet(x)    

class EarlyStopping:
    def __init__(self, patience=10, verbose=False, delta=0):
        """
        Args:
            patience (int): Patience value, specifies how many epochs without improvement before stopping training.
            verbose (bool): If True, prints a message indicating early stopping.
            delta (float): Minimum change to be considered as improvement, only improvements greater than this are considered valid.
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta

    def __call__(self, val_loss, model):
        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                print(f"EarlyStopping counter: {self.counter} out of {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        """Saves model when validation loss decreases."""
        if self.verbose:
            print(f"Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}). Saving model ...")
        torch.save(model.state_dict(), "checkpoint.pt")
        self.val_loss_min = val_loss


class CustomNet(nn.Module):
    # Parameter settings
    # dist_mode:[0,1,2,3,4] represents distance in metric space: 0 for Euclidean; 1 for Ball; 2 for spd; 3 for Hellinger; 4 for TV.
    # device:['cpu', 'cuda:0', 'cuda:1']
     
    def __init__(self, architecture='FNN', optimizer_name='Adam', blambda=0.01, device='cpu', output_dim=10,
                p_dim=10, batch_size=100, iter_num=100, dist_mode=0, lr=1e-3):
        
        super(CustomNet, self).__init__()
        self.device = torch.device(device)
        self.output_dim = output_dim
        self.blambda = blambda
        self.p_dim = p_dim
        self.batch_size = batch_size
        self.iter_num = iter_num
        self.dist_mode = dist_mode
        self.lr = lr
        self.arch = architecture

        # Build network based on chosen architecture
        if architecture == 'FNN':
            self.model = self.build_fnn()
        elif architecture == 'CNN':
            self.model = self.build_cnn()
        elif architecture == 'LeNet':
            self.model = self.build_lenet()
        elif architecture == 'ResNet':
            self.model = ResNet18(self.output_dim)
        elif architecture == 'DenseNet':
            self.model = CustomDenseNet121(self.output_dim)
        else:
            raise ValueError("Unsupported architecture type")


        # Initialize network parameter weights
        self.model.apply(init_weights)
        self.model.to(self.device)

        # Choose optimizer
        if optimizer_name == 'Adam':
            self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        elif optimizer_name == 'SGD':
            self.optimizer = optim.SGD(self.model.parameters(), lr=self.lr, momentum=0.9)
        else:
            raise ValueError("Unsupported optimizer type")
    
    def build_fnn(self):
        # Maximum width defined as 2**7 = 128
        max_width = 7 # 6
        init_width = 6 # 5
        min_width = 3 # 2
        middle = 0
        # Define network layers
        layers = [
            nn.Linear(self.p_dim, 2**init_width),  # Input layer to first hidden layer
            nn.ReLU()
        ]
        # Pyramid-shaped network
        for i in range(init_width, max_width):
            layers.append(nn.Linear(2**i, 2** (i+1)))
            layers.append(nn.ReLU())

        for _ in range(middle):
            layers.append(nn.Linear(2**max_width, 2**max_width))
            layers.append(nn.ReLU())
        
        for i in range(max_width, min_width, -1):
            layers.append(nn.Linear(2**i, 2**(i-1)))
            layers.append(nn.ReLU())
        # layers.append(nn.Linear(2**max_width, 2**min_width))
        # layers.append(nn.ReLU())

        # Last layer to output layer
        layers.append(nn.Linear(2**min_width, self.output_dim))

        # Create sequential model
        return nn.Sequential(*layers)    

    def build_cnn(self):
        # Example: A simple convolutional network
        return nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 128),
            nn.ReLU(),
            nn.Linear(128, self.output_dim)
        )

    def build_lenet(self): # for FER datasets
        layers = [nn.Conv2d(1, 6, kernel_size=5, padding=2),  # Output size: [6, 100, 100]
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),  # Output size: [6, 50, 50]
            nn.Conv2d(6, 16, kernel_size=5),  # Output size: [16, 46, 46]
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),  # Output size: [16, 23, 23]
            nn.Conv2d(16, 32, kernel_size=5, padding=2),  # Output size: [32, 23, 23]
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),  # Output size: [32, 11, 11]
            nn.Conv2d(32, 64, kernel_size=5, padding=2),  # Output size: [64, 11, 11]
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Flatten(),  # Flatten: [1600]
            nn.Linear(1600, 120),
            nn.ReLU(),
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Linear(84, self.output_dim)]
        return nn.Sequential(*layers)    

    def lenet_mnist(self): 
        layers = [nn.Conv2d(1, 6, kernel_size=5, padding=2),  #  [6, 28, 28]
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),  #  [6, 14, 14]
            nn.Conv2d(6, 16, kernel_size=5),  # [16, 10, 10]
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),  # [16, 5, 5]
            nn.Conv2d(16, 32, kernel_size=3, padding=1),  #  [32, 5, 5]
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),  #  [32, 2, 2]
            nn.Flatten(),  #  [128]
            nn.Linear(128, 120),
            nn.ReLU(),
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Linear(84, self.output_dim)]
        return nn.Sequential(*layers)
    
    def forward(self, x):
        return self.model(x)
    
    def train_step(self, data, target, loss_fn):
        if not isinstance(data, torch.Tensor):
            data = torch.tensor(data, dtype=torch.float32)
        if not isinstance(target, torch.Tensor):
            target = torch.tensor(target, dtype=torch.float32)
        # Training mode
        self.model.train()
        print("Net starts training")

        for epoch in range(self.iter_num):
            dataset = TensorDataset(data, target)
            dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
            # Total loss
            total_loss = 0
            for data_batch, target_batch in dataloader:
                # Transfer data batch to device
                data_batch, target_batch = data_batch.to(self.device), target_batch.to(self.device)
                self.optimizer.zero_grad()
                outputs = self.model(data_batch)
                loss = loss_fn(outputs, target_batch)
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()
            
            if self.arch=='DCNN':
                self.scheduler.step()
            avg_loss = total_loss / len(dataloader)
            if epoch%(self.iter_num/10)==0:
                print(f"epoch:{epoch}, avg_loss:",avg_loss)
        

    def train_step_early(self, data, target, val_data, val_target, loss_fn, patience=10):
        # Initialize early stopping object
        early_stopping = EarlyStopping(patience=patience, verbose=False)

        if not isinstance(data, torch.Tensor):
            data = torch.tensor(data, dtype=torch.float32)
            val_data = torch.tensor(val_data, dtype=torch.float32)
        if not isinstance(target, torch.Tensor):
            target = torch.tensor(target, dtype=torch.float32)
            val_target = torch.tensor(val_target, dtype=torch.float32)

        # Transfer validation set to device
        val_data, val_target = val_data.to(self.device), val_target.to(self.device)
        # Training mode
        self.model.train()
        print("Net starts training")
        
        for epoch in range(self.iter_num):
            dataset = TensorDataset(data, target)
            dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

            for data_batch, target_batch in dataloader:
                # Transfer data batch to device
                data_batch, target_batch = data_batch.to(self.device), target_batch.to(self.device)
                # Total loss
                total_loss = 0

                self.optimizer.zero_grad()
                outputs = self.model(data_batch)
                loss = loss_fn(outputs, target_batch)
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item() * self.batch_size
                
            self.scheduler.step()
            avg_loss = total_loss / len(dataloader)
            # print(f"epoch:{epoch}, train loss:",avg_loss)
            # Evaluate performance on validation set
            self.model.eval()
            with torch.no_grad():
                val_outputs = self.model(val_data)
                val_loss = loss_fn(val_outputs, val_target).item()

            # Call early stopping object
            early_stopping(val_loss, self.model)

            if early_stopping.early_stop:
                print("Early stopping")
                break

        # Load best model state
        self.model.load_state_dict(torch.load("checkpoint.pt"))

    def transform(self, test_data):
        self.model.eval()  # Set model to evaluation mode
        outputs = []

        with torch.no_grad():
            if not isinstance(test_data, torch.Tensor):
                test_data = torch.tensor(test_data, dtype=torch.float32)

            # Process data in batches
            total_data = test_data.size(0)
            for i in range(0, total_data, self.batch_size):
                # Process each batch
                batch_data = test_data[i:i+self.batch_size].to(self.device)
                batch_output = self.forward(batch_data)
                outputs.append(batch_output)

            # Concatenate outputs from all batches
            output = torch.cat(outputs, dim=0)

            if not isinstance(output, np.ndarray):
                # If not, try to convert to numpy.ndarray
                if isinstance(output, torch.Tensor):
                    # Ensure PyTorch tensor is moved to CPU and converted to NumPy array
                    output = output.cpu().numpy()
                else:
                    output = np.array(output)

        return output.astype(np.float32)


     #************************* Functions below are for computing mccov loss *****************************

    def dist(self,y1,y2,mode):     # Input y1 and y2 are both (n,n,d) tensors, representing calculating distances of (n,n) corresponding d-dimensional vectors
                                # When there is a new distance d, just calculate d(y1[n,n],y2[n,n]) according to (n,n) components
        # mode represents different distances, 0 for Euclidean
        if mode==0:
            distances = torch.norm(y1-y2, dim=2)
        # 1 for spherical distance arccos(y'y) if y is a one-dimensional vector
        elif mode==1:
            distances = torch.arccos(torch.einsum('ijk,ijk->ij', y1, y2))
        # 2 for spd_affine-invariant distance
        elif mode==2:
            distances = []
            for i in range(y1.shape[0]):
                dis = spd_dis(y1[i],y2[i])
                distances.append(dis)
            distances = torch.stack(distances)
        # 3 for Hellinger distance of discrete distributions
        elif mode==3:
            z1 = torch.sqrt(y1)
            z2 = torch.sqrt(y2)
            distances = torch.norm(z1-z2, dim=2)/np.sqrt(2)
        # 4 for TV distance of discrete distributions
        elif mode==4:
            distances = 0.5*torch.sum(torch.abs(y1-y2), dim=2)
        # 5 for indicator function of discrete variables
        elif mode==5:
            equal_elements = torch.eq(y1, y2)
            distances = torch.all(equal_elements, dim=2).float()
        return distances    
    
    def calculate_distances_and_sort_torch(self, X, Y, mode=0):
        d = X.dim()
        n = X.shape[0]
        mean_Y = torch.mean(Y, dim=0) if Y.shape[1] > 1 else torch.mean(Y)
         # Pre-allocate memory space
        Y_combined = torch.empty_like(Y)
        if d==2:
            # Batch compute distances
            expanded_X = X.unsqueeze(1).expand(-1, n, -1)
            distances = self.dist(expanded_X, expanded_X.transpose(0, 1), mode)
        elif d==3:    # Data is spd matrices at this time
            expanded_X = X.unsqueeze(1).expand(-1, n, -1, -1)
            distances = self.dist(expanded_X, expanded_X.transpose(0, 1), mode)

        # Batch sorting
        sorted_indices = torch.argsort(distances, dim=1)
        sorted_Y = torch.gather(Y.expand(n, -1, -1), 1, sorted_indices.unsqueeze(-1).expand(-1, -1, Y.shape[1]))

        # Batch subtract mean
        Y_combined = sorted_Y - mean_Y
    
        return Y_combined


    # Fast computation of MCCOV
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
        # Metric space distance mode 1 represents spherical distance
        mode = self.dist_mode
        yhat_sorted = self.calculate_distances_and_sort_torch(y, yhat, mode)  # Use PyTorch version of function
        num = 0.0  # Use float to maintain precision

        if yhat.ndim == 1:
            num = self.fast_cmp_torch(yhat_sorted, yhat)
        elif yhat.ndim == 2:
            q = yhat.shape[1]
            for i in range(q):
                num += self.fast_cmp_torch(yhat_sorted[:, :, i], yhat[:, i])
        return -1.0*num

    def mccov_loss(self, yhat, y):          # Acceleration method, can write centering once in training function instead of re-centering each time computing loss
        yhatdim = yhat.shape[1]
        cov_loss =  torch.norm((torch.cov(yhat.T).reshape(yhatdim, yhatdim) - torch.eye(yhatdim).to(self.device)), p='fro')**2
        mccov = self.mccov_value(yhat, y)
        return mccov + self.blambda * cov_loss


    #************************* Functions above are for computing mccov loss *****************************

    def evaluate(self, test_loader):
        self.model.eval()  # Set model to evaluation mode
        total_correct = 0
        total_samples = 0
        
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(self.device), target.to(self.device)
                outputs = self.model(data)
                _, predicted = torch.max(outputs.data, 1)
                total_samples += target.size(0)
                total_correct += (predicted == target).sum().item()
    
        accuracy = total_correct / total_samples
        print(f'Accuracy of the network on the test images: {accuracy * 100:.2f}%')
        return accuracy