import torch
import numpy as np

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class GCML_Loss(object):
    def __init__(self, param, k=5, push_scale=3):
        self.k = k
        self.push_scale = push_scale
        self.n_clusters = param['n_clusters']
        
        self.data_dis = torch.zeros((param['data_num'], self.k), device=device)
        self.out_dis = torch.zeros((param['data_num'], self.k), device=device)
        
    def t_distribute(self, dis, alpha=1.0):

        numerator = 1.0 / (1.0 + (dis**2 / alpha))
        power = float(alpha + 1.0) / 2
        numerator = numerator ** power
        t_dist = (numerator.t() / torch.sum(numerator, 1)).t()

        return t_dist

    def kNNGraph(self, data):

        n_samples = data.shape[0]

        x = data.to(device)
        y = data.to(device)
        m, n = x.size(0), y.size(0)
        
        xx = torch.pow(x, 2).sum(1, keepdim=True).expand(m, n)
        yy = torch.pow(y, 2).sum(1, keepdim=True).expand(n, m).t()
        
        dist = xx + yy
        dist.addmm_(1, -2, x, y.t())
        d = dist.clamp(min=1e-8).sqrt()

        kNN_mask = torch.zeros((n_samples, n_samples), device=device)
        s_, indices = torch.sort(d, dim=1)
        kNN_mask.scatter_(1, indices[:, 1:self.k+1], 1)

        return d, kNN_mask.bool()

    # Imposing local isometry within each manifold
    def update(self, data, out, y_pred):
        index_lists = torch.tensor(np.where(y_pred == 0)).to(device).view(-1)
        data_dis, _ = self.kNNGraph(data[y_pred == 0])
        out_dis, out_knn = self.kNNGraph(out[y_pred == 0])
        data_dis_masks = data_dis[out_knn].view(-1, self.k)
        out_dis_masks = out_dis[out_knn].view(-1, self.k)

        for i in range(1, self.n_clusters):
            index_list = torch.tensor(np.where(y_pred == i)).to(device).view(-1)
            data_dis, _ = self.kNNGraph(data[y_pred == i])
            out_dis, out_knn = self.kNNGraph(out[y_pred == i])
            data_dis_mask = data_dis[out_knn].view(-1, self.k)
            out_dis_mask = out_dis[out_knn].view(-1, self.k)
            
            data_dis_masks = torch.cat((data_dis_masks, data_dis_mask), 0)
            out_dis_masks = torch.cat((out_dis_masks, out_dis_mask), 0)
            index_lists = torch.cat((index_lists, index_list), 0)

        _, idx2 = torch.sort(index_lists)
        data_dis_masks = torch.index_select(data_dis_masks, 0, idx2)
        out_dis_masks = torch.index_select(out_dis_masks, 0, idx2)

        self.data_dis = data_dis_masks / torch.sqrt(torch.tensor(float(data.shape[1])))
        self.out_dis = out_dis_masks / torch.sqrt(torch.tensor(float(out.shape[1])))

    def calculate(self, centers_data, centers_out):

        centers_data, _ = self.kNNGraph(centers_data)
        centers_out, _ = self.kNNGraph(centers_out)
        loss_push = torch.norm(centers_out - centers_data * self.push_scale) / self.n_clusters

        data_dis_mask = self.data_dis
        out_dis_mask = self.out_dis
        loss_iso = torch.norm(data_dis_mask - out_dis_mask)

        return loss_push, loss_iso
    