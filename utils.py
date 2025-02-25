import numpy as np
from sklearn.model_selection import train_test_split
import os
import torch
import json

# define get_dataset function to split the data into training and validation sets
# use the np.float32 dtype to match the data type of the PyTorch tensors
def get_dataset(root="data", random_state=42):
    """Returns the training and validation datasets."""
    # Load the data
    x = np.load(os.path.join(root, 'x_data.npy')).astype(np.float32)
    y = np.load(os.path.join(root, 'y_data.npy')).astype(np.float32)
    z = np.load(os.path.join(root, 'z_data.npy')).astype(np.float32)

    # Split the data into training and validation sets
    # train : val : test = 6:2:2
    train_x, test_x, train_y, test_y, train_z, test_z = train_test_split(x, y, z, test_size=0.2, random_state=random_state)
    train_x, val_x, train_y, val_y, train_z, val_z = train_test_split(train_x, train_y, train_z, test_size=0.25, random_state=random_state)

    return (train_x, test_x, val_x), (train_y, test_y, val_y), (train_z, test_z, val_z)

def get_mean_std(data_dir="data"):
    x_mean = np.load(os.path.join(data_dir, "x_mean.npy")).reshape(1, -1)
    x_std = np.load(os.path.join(data_dir, "x_std.npy")).reshape(1, -1)
    y_mean = np.load(os.path.join(data_dir, "y_mean.npy")).reshape(1, -1)
    y_std = np.load(os.path.join(data_dir, "y_std.npy")).reshape(1, -1)
    z_mean = np.load(os.path.join(data_dir, "z_mean.npy")).reshape(1, -1)
    z_std = np.load(os.path.join(data_dir, "z_std.npy")).reshape(1, -1)
    return x_mean, y_mean, z_mean, x_std, y_std, z_std

#%%
def get_loader(data_type:str, data_dir:str="data", batch_size=32, shuffle_test=False, num_workers=0):
    '''
    Returns the training and validation data loaders. train : val : test = 6:2:2
    '''

    print("Loading data...")
    # Get the training and validation datasets
    data_dir = os.path.join(data_dir, data_type)
    (train_x, test_x, val_x), (train_y, test_y, val_y), (train_z, test_z, val_z) = get_dataset(data_dir)
    
    x_mean, y_mean, z_mean, x_std, y_std, z_std = get_mean_std(data_dir)
    train_x = (train_x - x_mean) / x_std
    test_x = (test_x - x_mean) / x_std
    val_x = (val_x - x_mean) / x_std
    train_y = (train_y - y_mean) / y_std
    test_y = (test_y - y_mean) / y_std
    val_y = (val_y - y_mean) / y_std
    train_z = (train_z - z_mean) / z_std
    test_z = (test_z - z_mean) / z_std
    val_z = (val_z - z_mean) / z_std

    # Create TensorDatasets
    train_dataset = torch.utils.data.TensorDataset(torch.from_numpy(train_x), torch.from_numpy(train_y), torch.from_numpy(train_z))
    test_dataset =  torch.utils.data.TensorDataset(torch.from_numpy(test_x), torch.from_numpy(test_y), torch.from_numpy(test_z))
    val_dataset =   torch.utils.data.TensorDataset(torch.from_numpy(val_x), torch.from_numpy(val_y), torch.from_numpy(val_z))
    print("Loaded.")

    # Create DataLoaders
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=shuffle_test, num_workers=num_workers)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=shuffle_test, num_workers=num_workers)

    return train_loader, test_loader, val_loader

class Statistics:
    '''
    If momentum = None, calculate the average of all the values.
    Else if momentum in (0, 1), calculate the exponential moving average.
    '''
    def __init__(self, momentum:float=None):
        self.n = 0
        self.avg = 0.0
        self.last = 0.0
        self.momentum = momentum

    def update(self, x, num:int=1):
        with torch.no_grad():
            if self.momentum is None:
                self.avg = self.avg * (self.n / (self.n + num)) + x * num / (self.n + num)
                self.n += num
                self.last = x
            else:
                self.avg = self.avg * (1-self.momentum) + x * self.momentum

    @staticmethod
    def get_statistics(k, **kwargs):
        return [Statistics(**kwargs) for _ in range(k)]


class RSquaredStatistics(Statistics):
    '''
    Require feature_mean to be shape consistent with y_true.
    '''
    def __init__(self, feature_mean, momentum:float=None):
        super().__init__(momentum)
        self.ss_res = 0.0
        self.ss_tot = 0.0
        self.feature_mean = feature_mean

    def update(self, y_true, y_pred, num:int=1):
        with torch.no_grad():
            ss_res = ((y_true - y_pred) ** 2).sum(dim=0) # sum over batch index
            ss_tot = ((y_true - self.feature_mean) ** 2).sum(dim=0) # sum over batch index
            self.ss_res = self.ss_res * (self.n / (self.n + num)) + ss_res * num / (self.n + num)
            self.ss_tot = self.ss_tot * (self.n / (self.n + num)) + ss_tot * num / (self.n + num)
            self.n += num
            self.avg = (1 - self.ss_res / self.ss_tot).mean().item()

    @staticmethod
    def get_statistics(k, **kwargs):
        return [RSquaredStatistics(**kwargs) for _ in range(k)]
    
def write_perf_to_json(perf_dict, save_root, filename:str="log.json"):
    filepath = os.path.join(save_root, filename)
    with open(filepath, "w") as f:
        json.dump(perf_dict, f, indent=4)

def load_perf_from_json(load_root, filename:str="log.json"):
    filepath = os.path.join(load_root, filename)
    if not os.path.isfile(filepath):
        print(filepath, "does not exist!")
        return None
    with open(filepath, "r") as f:
        perf_dict = json.load(f)
    return perf_dict
