import os 
os.environ["CUDA_VISIBLE_DEVICES"] = "7"

import torch
import numpy as np
from scipy.spatial.distance import pdist, squareform

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def GetIndicator(data, latent):

    if torch.is_tensor(data):
        data = data.detach().cpu().numpy()
    if torch.is_tensor(latent):
        latent = latent.detach().cpu().numpy()

    calc = MeasureCalculator(data, latent)

    rmse_local = []
    mrreZX = []
    mrreXZ = []
    cont = []
    trust = []

    for k in range(3, 8):
        rmse_local.append(calc.local_rmse(k=k))
        mrreZX.append(calc.mrre(k)[0])
        mrreXZ.append(calc.mrre(k)[1])
        cont.append(calc.continuity(k))
        trust.append(calc.trustworthiness(k))

    indicator = {}
    indicator['RRE'] = (np.mean(mrreXZ) + np.mean(mrreZX)) / 2.0
    indicator['Cont'] = np.mean(cont)
    indicator['Trust'] = np.mean(trust)
    indicator['RMSE'] = calc.rmse()
    indicator['LGD'] = np.mean(rmse_local)

    return indicator

class MeasureCalculator():
    def __init__(self, Xi, Zi, k_max=51):
        if torch.is_tensor(Xi):
            self.X = Xi.detach().cpu().numpy()
            self.Z = Zi.detach().cpu().numpy()
        else:
            self.X = Xi
            self.Z = Zi
        
        self.init(k_max)

    def init(self, k_max):
        
        batch_num = 2000
        data_num = self.X.shape[0] % batch_num
        
        if data_num == 0:
            data_num = batch_num
            batch = self.X.shape[0]//batch_num - 1
        else:
            data_num = data_num
            batch = self.X.shape[0]//batch_num
            
        self.pairwise_X = self.kNNGraph(self.X[:data_num], self.X)
        self.pairwise_Z = self.kNNGraph(self.Z[:data_num], self.Z)

        self.neighbours_X, self.ranks_X = self._neighbours_and_ranks(self.pairwise_X, k_max)
        self.neighbours_Z, self.ranks_Z = self._neighbours_and_ranks(self.pairwise_Z, k_max)

        self.pairwise_X = self.pairwise_X.cpu().numpy()
        self.pairwise_Z = self.pairwise_Z.cpu().numpy()
        self.neighbours_X = self.neighbours_X.cpu().numpy()
        self.neighbours_Z = self.neighbours_Z.cpu().numpy()
        self.ranks_X = self.ranks_X.cpu().numpy()
        self.ranks_Z = self.ranks_Z.cpu().numpy()

        torch.cuda.empty_cache()

        for i in range(0, batch):
            self.pairwise_Xs = self.kNNGraph(self.X[data_num+batch_num*i:data_num+batch_num*(i+1)], self.X)
            self.pairwise_Zs = self.kNNGraph(self.Z[data_num+batch_num*i:data_num+batch_num*(i+1)], self.Z)

            self.neighbours_Xs, self.ranks_Xs = self._neighbours_and_ranks(self.pairwise_Xs, k_max)
            self.neighbours_Zs, self.ranks_Zs = self._neighbours_and_ranks(self.pairwise_Zs, k_max)

            self.pairwise_X = np.concatenate((self.pairwise_X, self.pairwise_Xs.cpu().numpy()), axis=0)
            self.pairwise_Z = np.concatenate((self.pairwise_Z, self.pairwise_Zs.cpu().numpy()), axis=0)
            self.neighbours_X = np.concatenate((self.neighbours_X, self.neighbours_Xs.cpu().numpy()), axis=0)
            self.neighbours_Z = np.concatenate((self.neighbours_Z, self.neighbours_Zs.cpu().numpy()), axis=0)
            self.ranks_X = np.concatenate((self.ranks_X, self.ranks_Xs.cpu().numpy()), axis=0)
            self.ranks_Z = np.concatenate((self.ranks_Z, self.ranks_Zs.cpu().numpy()), axis=0)

            torch.cuda.empty_cache()
            print(self.pairwise_X.shape, self.neighbours_X.shape, self.ranks_X.shape)

    def kNNGraph(self, x, y):

        x = torch.tensor(x).to(device)
        y = torch.tensor(y).to(device)
        m, n = x.size(0), y.size(0)
        
        xx = torch.pow(x, 2).sum(1, keepdim=True).expand(m, n)
        yy = torch.pow(y, 2).sum(1, keepdim=True).expand(n, m).t()
        
        dist = xx + yy
        dist.addmm_(1, -2, x, y.t())
        d = dist.clamp(min=1e-8).sqrt()

        return d

    def _neighbours_and_ranks(self, distances, k):

        _, indices = torch.sort(distances, dim=1)
        neighbourhood = indices[:, 1:k+1]
        _, ranks = torch.sort(indices, dim=1)

        return neighbourhood, ranks

    def get_X_neighbours_and_ranks(self, k):
        return self.neighbours_X[:, :k], self.ranks_X

    def get_Z_neighbours_and_ranks(self, k):
        return self.neighbours_Z[:, :k], self.ranks_Z

    def rmse(self):
        n = self.pairwise_X.shape[0]
        sum_of_squared_differences = np.square(self.pairwise_X - self.pairwise_Z).sum()

        return np.sqrt(sum_of_squared_differences / n**2)
    
    def local_rmse(self, k):
        X_neighbors, _ = self.get_X_neighbours_and_ranks(k)

        mses = []
        n = self.pairwise_X.shape[0]
        for i in range(n):
            x = self.X[X_neighbors[i]]
            z = self.Z[X_neighbors[i]]
            d1 = np.sqrt(np.square(x - self.X[i]).sum(axis=1))/np.sqrt(self.X.shape[1])
            d2 = np.sqrt(np.square(z - self.Z[i]).sum(axis=1))/np.sqrt(self.Z.shape[1])
            mse = np.sum(np.square(d1 - d2))
            mses.append(mse)

        return np.sqrt(np.sum(mses)/(k*n))
        
    def _trustworthiness(self, X_neighbourhood, X_ranks, Z_neighbourhood, Z_ranks, n, k):

        result = 0.0

        for row in range(X_ranks.shape[0]):
            missing_neighbours = np.setdiff1d(Z_neighbourhood[row], X_neighbourhood[row])

            for neighbour in missing_neighbours:
                result += (X_ranks[row, neighbour] - k)

        return 1 - 2 / (n * k * (2 * n - 3 * k - 1)) * result

    def trustworthiness(self, k):

        X_neighbourhood, X_ranks = self.get_X_neighbours_and_ranks(k)
        Z_neighbourhood, Z_ranks = self.get_Z_neighbours_and_ranks(k)
        n = self.pairwise_X.shape[0]

        return self._trustworthiness(X_neighbourhood, X_ranks, Z_neighbourhood, Z_ranks, n, k)

    def continuity(self, k):

        X_neighbourhood, X_ranks = self.get_X_neighbours_and_ranks(k)
        Z_neighbourhood, Z_ranks = self.get_Z_neighbours_and_ranks(k)
        n = self.pairwise_X.shape[0]

        return self._trustworthiness(Z_neighbourhood, Z_ranks, X_neighbourhood, X_ranks, n, k)

    def mrre(self, k):

        X_neighbourhood, X_ranks = self.get_X_neighbours_and_ranks(k)
        Z_neighbourhood, Z_ranks = self.get_Z_neighbours_and_ranks(k)

        n = self.pairwise_X.shape[0]

        mrre_ZX = 0.0
        for row in range(n):
            for neighbour in Z_neighbourhood[row]:
                rx = X_ranks[row, neighbour]
                rz = Z_ranks[row, neighbour]
                mrre_ZX += abs(rx - rz) / rz

        mrre_XZ = 0.0
        for row in range(n):
            for neighbour in X_neighbourhood[row]:
                rx = X_ranks[row, neighbour]
                rz = Z_ranks[row, neighbour]
                mrre_XZ += abs(rx - rz) / rx

        C = n * sum([abs(2*j - n) / j for j in range(1, k+1)])

        return mrre_ZX / C, mrre_XZ / C



