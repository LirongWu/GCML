import numpy as np
import pandas as pd
import scipy.io as sio

import torch
from torch.utils.data import Dataset


def load_mnist(path='./data/MNIST/mnist.npz', start_idx=0, data_num=70000):
    data_file = np.load(path)
    x_train, y_train, x_test, y_test = data_file['x_train'], data_file['y_train'], data_file['x_test'], data_file['y_test']
    data_file.close()

    x = np.concatenate((x_train, x_test)).astype(np.float32)
    y = np.concatenate((y_train, y_test)).astype(np.int32)

    x = x.reshape((x.shape[0], -1)) / 255.
    print('MNIST samples', x.shape)

    return x[start_idx:start_idx+data_num], y[start_idx:start_idx+data_num]

def load_usps(path='./data/USPS/usps_resampled.mat', start_idx=0, data_num=9298):
    data = sio.loadmat(path)
    x_train, y_train, x_test, y_test = data['train_patterns'].T, data['train_labels'].T, data['test_patterns'].T, data['test_labels'].T
    
    y_train = [np.argmax(l) for l in y_train]
    y_test = [np.argmax(l) for l in y_test]
    x = np.concatenate((x_train, x_test)).astype(np.float32)
    y = np.concatenate((y_train, y_test)).astype(np.int32)

    x = (x.reshape((x.shape[0], -1)) + 1.0) / 2.0
    print('USPS samples', x.shape)

    return x[start_idx:start_idx+data_num], y[start_idx:start_idx+data_num]

def load_fashionmnist(path='./data/Fashion-MNIST/', start_idx=0, data_num=70000):
    x = np.load(path + 'data.npy').astype(np.float32)
    y = np.load(path + 'labels.npy').astype(np.int32)

    x = x.reshape((x.shape[0], -1))
    print('FashionMNIST samples', x.shape)

    return x[start_idx:start_idx+data_num], y[start_idx:start_idx+data_num]

def load_reuters10k(path='./data/Reuters-10k/reuters-10k.npy', start_idx=0, data_num=10000):
    data = np.load(path, allow_pickle=True).item()

    x = data['data']
    y = data['label']
    x = x.reshape((x.shape[0], -1)).astype(np.float32)
    y = y.reshape((y.shape[0])).astype(np.int32)
    print(('REUTERSIDF10K samples', x.shape))

    return x[start_idx:start_idx+data_num], y[start_idx:start_idx+data_num]

def load_har(path='./data/HAR/', start_idx=0, data_num=10000):
    x_train = pd.read_csv(path + 'train/X_train.txt', sep=r'\s+', header=None)
    y_train = pd.read_csv(path + 'train/y_train.txt', header=None)
    x_test = pd.read_csv(path + 'test/X_test.txt', sep=r'\s+', header=None)
    y_test = pd.read_csv(path + 'test/y_test.txt', header=None)

    x = np.concatenate((x_train, x_test)).astype(np.float32)
    y = np.concatenate((y_train, y_test)).astype(np.int32)
    y = y - 1
    y = y.reshape((y.size,))
    print(('HAR samples', x.shape))

    return x[start_idx:start_idx+data_num], y[start_idx:start_idx+data_num]

def load_pendigits(path='./data/Pendigits/', start_idx=0, data_num=10992):
    with open(path + 'pendigits.tra') as file:
        data = file.readlines()
    data = [list(map(float, line.split(','))) for line in data]
    data = np.array(data).astype(np.float32)
    data_train, labels_train = data[:, :-1], data[:, -1]

    with open(path + '/pendigits.tes') as file:
        data = file.readlines()
    data = [list(map(float, line.split(','))) for line in data]
    data = np.array(data).astype(np.float32)
    data_test, labels_test = data[:, :-1], data[:, -1]

    x = np.concatenate((data_train, data_test)).astype('float32')
    y = np.concatenate((labels_train, labels_test))
    x /= 100.
    y = y.astype('int')

    print('Pendigits samples', x.shape)

    return x[start_idx:start_idx+data_num], y[start_idx:start_idx+data_num]


class Dataset(Dataset):

    def __init__(self, start_idx, data_num, datasets='MNIST'):
        if datasets == 'MNIST':
            self.x, self.y = load_mnist(start_idx=start_idx, data_num=data_num)
        if datasets == 'USPS':
            self.x, self.y = load_usps(start_idx=start_idx, data_num=data_num)
        if datasets == 'Fashion-MNIST':
            self.x, self.y = load_fashionmnist(start_idx=start_idx, data_num=data_num)
        if datasets == 'Reuters-10k':
            self.x, self.y = load_reuters10k(start_idx=start_idx, data_num=data_num)
        if datasets == 'HAR':
            self.x, self.y = load_har(start_idx=start_idx, data_num=data_num)
        if datasets == 'Pendigits':
            self.x, self.y = load_pendigits(start_idx=start_idx, data_num=data_num)

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, idx):
        return torch.from_numpy(np.array(self.x[idx])), torch.from_numpy(np.array(self.y[idx])), torch.from_numpy(np.array(idx))