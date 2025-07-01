import torch

from torch.utils.data import Dataset
import os
import numpy as np

import random
# from .augmentations import DataTransform
#---------------------------------------------------#
#   设置种子
#---------------------------------------------------#
#---------------------------------------------------#
#   设置Dataloader的种子
#---------------------------------------------------#
def worker_init_fn(worker_id, seed=11):
    worker_seed = seed + worker_id
    # random.seed(worker_seed)
    np.random.seed(worker_seed)
    # torch.manual_seed(worker_seed)


def slide_window(data,window_size):
    X_list, Y_list = [], []
    num_of_seq = data.shape[0]-window_size
    for j in range(num_of_seq):
        X = data[j:j+window_size]
        Y = data[j+window_size:j+window_size+1]
        X_list.append(X)
        Y_list.append(Y)
    return torch.stack(X_list), torch.stack(Y_list)
class Load_Dataset(Dataset):
    # Initialize your data, download, etc.
    def __init__(self, input, label):
        super(Load_Dataset, self).__init__()
        # self.training_mode = training_mode

       
        self.X = input
        # self.X_close = dataset["close"]
        self.Y = label
        self.len=len(input)

    def __getitem__(self, index):
        return self.X[index], self.Y[index]
        

    def __len__(self):
        return self.len

def data_generator(data_path, configs, input_len=None):
    origin_data = torch.load('data/mobile_data.pt',weights_only=False)
    days,time_intervals,n_nodes = origin_data.size()
    train_dataset = origin_data[:configs['train_days']]
    valid_dataset = origin_data[configs['train_days']:configs['valid_days']+configs['train_days']]
    test_dataset = origin_data[configs['valid_days']+configs['train_days']:]
    if input_len==None:
        input_len = configs['input_len_period']
    train_dataset_X, train_dataset_Y = slide_window(train_dataset.reshape(configs['train_days']*configs['num_time_interval'], configs['n_nodes']), input_len)
    valid_dataset_X, valid_dataset_Y = slide_window(valid_dataset.reshape(configs['valid_days']*configs['num_time_interval'], configs['n_nodes']), input_len)
    test_dataset_X, test_dataset_Y = slide_window(test_dataset.reshape((days-configs['train_days']-configs['valid_days'])*configs['num_time_interval'], configs['n_nodes']), input_len)
    train_dataset = Load_Dataset(train_dataset_X, train_dataset_Y)
    valid_dataset = Load_Dataset(valid_dataset_X, valid_dataset_Y)
    test_dataset = Load_Dataset(test_dataset_X, test_dataset_Y)

    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=configs['batch_size'],
                                               shuffle=True, drop_last=configs['drop_last'],worker_init_fn=worker_init_fn,
                                               num_workers=0)
    valid_loader = torch.utils.data.DataLoader(dataset=valid_dataset,  batch_size=configs['batch_size'],
                                               shuffle=False, drop_last=configs['drop_last'],
                                               num_workers=0)

    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,  batch_size=configs['batch_size'],
                                               shuffle=False, drop_last=configs['drop_last'],
                                              num_workers=0)
    return train_loader, valid_loader, test_loader