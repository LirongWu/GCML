import umap
import torch
import numpy as np
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from matplotlib import pyplot as plt
from scipy.optimize import linear_sum_assignment as linear_assignment
from sklearn.metrics.cluster import normalized_mutual_info_score as nmi_score

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# Set seeds to ensure reproducibility
def SetSeed(seed):
    SEED = seed
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    np.random.seed(SEED)

# Add Gaussian noise for denoising autoencoder
def add_noise(data, noise_level):

    noise = torch.randn(data.size()).to(device) * noise_level
    noisy_data = data + noise

    return noisy_data

# Matrix of Euclidean distances
def kNNGraph(data):

    x = data.to(device)
    y = data.to(device)
    m, n = x.size(0), y.size(0)
    
    xx = torch.pow(x, 2).sum(1, keepdim=True).expand(m, n)
    yy = torch.pow(y, 2).sum(1, keepdim=True).expand(n, m).t()
    
    dist = xx + yy
    dist.addmm_(1, -2, x, y.t())
    d = dist.clamp(min=1e-8).sqrt()

    return d

# Calculating the accuracy of clustering
def cluster_acc(y_true, y_pred):
    y_true = y_true.astype(np.int64)
    assert y_pred.size == y_true.size

    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)

    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1

    ind = linear_assignment(w.max() - w)
    ind = np.array(ind).T

    return sum([w[i, j] for i, j in ind]) * 1.0 / y_pred.size

# Calculating the accuracy of rank-preservation
def rank_acc(data_centers, out_centers):
    data_dis = kNNGraph(data_centers)
    out_dis = kNNGraph(out_centers)

    data_sort = np.argsort(data_dis.detach().cpu().numpy())
    out_sort = np.argsort(out_dis.detach().cpu().numpy())

    acc = np.sum(np.where(data_sort == out_sort, 1, 0)) / (data_centers.shape[0] ** 2)

    return acc
    
# Visualize clustering results and save intermediate data
def visualize(data, labels, y_pred, model, epoch, path, param):
    hidden = model.autoencoder.encode(data).detach().cpu().numpy()	
    np.save("{}data/{}_e{:03d}_hidden.npy".format(path, param['dataset_type'], epoch), hidden)
    np.save("{}data/{}_e{:03d}_centers.npy".format(path, param['dataset_type'], epoch), model.cluster_centers.detach().cpu().numpy())
    np.save("{}data/{}_e{:03d}_y_pred.npy".format(path, param['dataset_type'], epoch), y_pred)

    reducer = umap.UMAP(n_neighbors=5, min_dist=0.7,  metric='correlation')
    x_embedded = reducer.fit_transform(hidden)
    
    fig = plt.figure()
    ax = plt.subplot(111)
    plt.scatter(x_embedded[:,0], x_embedded[:,1], c=labels, s=1, cmap='rainbow_r')
    plt.axis('equal')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    plt.xticks([])
    plt.yticks([])
    plt.tight_layout()

    output = model(data)[1].argmax(1).detach().cpu().numpy()
    acc = cluster_acc(labels, output)
    nmi = nmi_score(labels, output)
    fig.savefig('{}pics/{}_e{:03d}_acc{}_nmi{}.png'.format(path, param['dataset_type'], epoch, round(acc, 4), round(nmi, 4)))
    plt.close(fig)

    if epoch == 0:
        np.save("{}data/{}_input.npy".format(path, param['dataset_type']), data.cpu().numpy())
        np.save("{}data/{}_labels.npy".format(path, param['dataset_type']), labels)

# Initializing the clustering centers using K-means
def Clustering(data, model, hidden_num, data_num, clu_num):

    hidden = model.autoencoder.encode(data[:data_num])
    x_embedded = TSNE(n_components=2).fit_transform(hidden.detach().cpu().numpy())
    kmeans = KMeans(n_clusters=clu_num, random_state=0, n_init=20).fit(x_embedded)
    y_pred = kmeans.predict(x_embedded)

    cluster_centers = torch.zeros((clu_num, hidden_num))
    for i in range(clu_num):
        cluster_centers[i, :] = torch.mean(hidden[y_pred == i], 0)

    return y_pred, cluster_centers