import os
import json
import warnings
import argparse
import numpy as np
from sklearn.metrics.cluster import normalized_mutual_info_score as nmi_score

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from utils import *
from model import *
from dataset import *
from loss import *

warnings.filterwarnings("ignore", category=Warning)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# Pretrain the model with denoising autoencoder
def pretrain(model, num_epochs):

    train_loader = DataLoader(dataset, batch_size=param['batch_size'], shuffle=True)
    if param['dataset_type'] == "Reuters-10k":
        optimizer = torch.optim.SGD(model.parameters(), lr=1e-1, momentum=0.99)
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=param['lr'])
    
    for epoch in range(num_epochs):
        total_loss = 0

        for batch_idx, (x, _, _) in enumerate(train_loader):
            x = x.to(device)
            x_noise = add_noise(x, param['noise'])
            x_rec = model(x_noise)
            loss = F.mse_loss(x_rec, x)
            total_loss += loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print('\033[0;30;46mEpoch: [{}/{}], MSE_loss: {:.8f}\033[0m'.format(epoch + 1, num_epochs, total_loss / (batch_idx + 1)))

    # torch.save(model.state_dict(), "./model/{}/ae_model.pkl".format(param['name']))


def train(model, num_epochs, path):

    data = torch.Tensor(dataset.x).to(device)
    labels = dataset.y
    train_loader = DataLoader(dataset, batch_size=param['batch_size'], shuffle=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=param['lr'])
    optimizer_iso = torch.optim.Adam(model.parameters(), lr=param['lr'])
    Loss = GCML_Loss(param)

    # Initial clustering center
    _, cluster_centers = Clustering(data, model, param['hidden'], param['clu_num'], param['n_clusters'])
    model.cluster_centers.data = torch.tensor(cluster_centers).to(device)

    model.train()
    for epoch in range(num_epochs):
        total_loss = [0, 0, 0, 0, 0]

        # Update target distribution (p-distribution)
        _, q = model(data)
        p = model.target_distribution(q).detach()
        hidden = model.autoencoder.encode(data)
        y_pred = q.argmax(1).detach().cpu().numpy()
        Loss.update(data, hidden, y_pred)

        centers_data = torch.zeros((param['n_clusters'], data.shape[1])).to(device)
        centers_out = torch.zeros((param['n_clusters'], param['hidden'])).to(device)
        for i in range(param['n_clusters']):
            centers_data[i, :] = torch.mean(data[y_pred == i], 0)
            centers_out[i, :] = torch.mean(hidden[y_pred == i], 0)

        # Output visualization results once every 10 epochs
        if (epoch+1) % 50 == 0 or epoch == 0:
            visualize(data, labels, y_pred, model, epoch, path, param)

        for batch_idx, (x, _, idx) in enumerate(train_loader):
            x = x.to(device)
            x_rec, q = model(x)
            hidden = model.autoencoder.encode(x)

            loss_rec = F.mse_loss(x_rec, x)
            loss_cluster = F.kl_div(q.log(), p[idx])
            loss_push, loss_iso = Loss.calculate(centers_data, model.cluster_centers)
            loss_align = torch.sum(torch.norm(centers_out - model.cluster_centers, dim=1))  / param['n_clusters']

            total_loss[0] += loss_rec.item()
            total_loss[1] += loss_cluster.item()
            total_loss[2] += loss_push.item()
            total_loss[3] += loss_iso.item()
            total_loss[4] += loss_align.item()

            # Update L_ae, L_rec, L_rank
            if epoch < param['start_time']:
                loss = param['ratio'][0] * loss_cluster + loss_rec
            else:
                alpha = max(0, (1-(epoch-param['start_time'])/(num_epochs/2-param['start_time'])))
                loss = (param['ratio'][0] * loss_cluster + loss_rec) * alpha + param['ratio'][2] * loss_push / (param['data_num'] // param['batch_size'])

            optimizer.zero_grad()
            loss.backward(retain_graph=True)
            optimizer.step()

        # Update L_iso, L_align
        if epoch >= param['start_time']:
            optimizer_iso.zero_grad()
            loss_com = min(1, (epoch-param['start_time'])/(num_epochs/2)) * param['ratio'][1] * loss_iso + param['ratio'][3] * loss_align
            loss_com.backward()
            optimizer_iso.step()

        # Output evaluation results (ACC/NMI) every 10 epochs
        if (epoch+1) % 10 == 0 or epoch == 0:
            output = model(data)[1].argmax(1).detach().cpu().numpy()
            acc = cluster_acc(labels, output)
            nmi = nmi_score(labels, output)

            x_embedded = model.autoencoder.encode(data).detach().cpu().numpy()
            kmeans = KMeans(n_clusters=param['n_clusters'], random_state=0, n_init=20).fit(x_embedded)
            y_pred_kmeans = kmeans.predict(x_embedded)
            acc_kmeans = cluster_acc(labels, y_pred_kmeans)

            print('\033[0;30;46mEpochs: [{}/{}], Acc: {} {}, NMI: {}\033[0m'.format(epoch + 1, num_epochs, acc, acc_kmeans, nmi))      
        else:
            print('\033[0;30;46mEpochs: [{}/{}], Loss: {}\033[0m'.format(epoch + 1, num_epochs, np.array(total_loss) / (batch_idx + 1)))
        # torch.save(model.state_dict(), "./model/{}/kl_model.pkl".format(param['name']))


def test(model):
    dataset = Dataset(param['start_idx'], 70000, param['dataset_type'])
    data = torch.Tensor(dataset.x[50000:]).to(device)
    labels = dataset.y[50000:]
    hidden = model.autoencoder.encode(data).detach().cpu().numpy()	
    reducer = umap.UMAP(n_neighbors=5, min_dist=0.7,  metric='correlation')
    x_embedded = reducer.fit_transform(hidden)
    
    fig = plt.figure()
    ax = plt.subplot(111)
    plt.scatter(x_embedded[:,0], x_embedded[:,1], c=labels, s=1, cmap='rainbow_r')

    output = model(data)[1].argmax(1).detach().cpu().numpy()
    acc = cluster_acc(labels, output)
    nmi = nmi_score(labels, output)
    fig.savefig('{}pics/{}_test_acc{}_nmi{}.png'.format(path, param['dataset_type'], round(acc, 4), round(nmi, 4)))
    plt.close(fig)

    
if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='train', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--dataset_type', default='MNIST-full', type=str, choices=['MNIST-full', 'MNIST-test', 'USPS', 'Fashion-MNIST', 'Reuters-10k', 'HAR', 'Pendigits'])
    parser.add_argument('--name', default='DEC', type=str)
    parser.add_argument('--parampath', default='None', type=str)
    parser.add_argument('--start_idx', default=0, type=int)
    parser.add_argument('--data_num', default=70000, type=int)
    parser.add_argument('--clu_num', default=3000, type=int)

    parser.add_argument('--lr', default=1e-3, type=float)
    parser.add_argument('--batch_size', default=256, type=int)
    parser.add_argument('--n_clusters', default=10, type=int)
    parser.add_argument('--ae_epochs', default=200, type=int)
    parser.add_argument('--dec_epochs', default=300, type=int)
    parser.add_argument("--noise", default=0.2, type=float)
    parser.add_argument("--hidden", default=10, type=int)

    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--start_time", default=10, type=int)
    parser.add_argument("--ratio", default=[0.1, 1.0, 1.0, 1.0], type=float, nargs='+')
    parser.add_argument('--pretrain_ae', default=1, type=int)
    args = parser.parse_args()

    if args.dataset_type == 'MNIST-full':
        args.parampath = './param/MNIST-full.json'
    if args.dataset_type == 'MNIST-test':
        args.parampath = './param/MNIST-test.json'
    if args.dataset_type == 'USPS':
        args.parampath = './param/USPS.json'
    if args.dataset_type == 'Fashion-MNIST':
        args.parampath = './param/Fashion-MNIST.json'
    if args.dataset_type == 'Reuters-10k':
        args.parampath = './param/Reuters-10k.json'
    if args.dataset_type == 'HAR':
        args.parampath = './param/HAR.json'
    if args.dataset_type == 'Pendigits':
        args.parampath = './param/Pendigits.json'
    if args.parampath is not 'None':
        jsontxt = open(args.parampath, 'r').read()
        param = json.loads(jsontxt)
    else:
        param = args.__dict__

    # param['seed'] = args.seed
    # param['name'] = args.name

    SetSeed(param['seed'])
    # if os.path.exists("./Model/{}/".format(param['name'])) is False:
    #    os.makedirs("./Model/{}/".format(param['name']))
    if os.path.exists("./plots/{}/pics/".format(param['name'])) is False:
        os.makedirs("./plots/{}/pics/".format(param['name']))
    if os.path.exists("./plots/{}/data/".format(param['name'])) is False:
        os.makedirs("./plots/{}/data/".format(param['name']))

    json.dump(param, open("./plots/{}/param.json".format(param['name']), 'a'), indent=2)
    dataset = Dataset(param['start_idx'], param['data_num'], param['dataset_type'])

    autoencoder = AutoEncoder(hidden=param['hidden'], input_size=dataset.x.shape[1]).to(device)
    if param['pretrain_ae']:
        if param['dataset_type'] == "Fashion-MNIST":
            autoencoder.load_state_dict(torch.load('./model/Fashion-MNIST/ae_model.pkl'))
        elif param['dataset_type'] == "HAR":
            autoencoder.load_state_dict(torch.load('./model/HAR/ae_model.pkl'))
        elif param['dataset_type'] == "Pendigits":
            autoencoder.load_state_dict(torch.load('./model/Pendigits/ae_model.pkl'))
        else:
            autoencoder.load_state_dict(torch.load('./model/MNIST/ae_model.pkl'))
    else:
        pretrain(model=autoencoder, num_epochs=param['ae_epochs'])
    
    dec = GCML(n_clusters=param['n_clusters'], hidden=param['hidden'], autoencoder=autoencoder).to(device)
    train(model=dec, num_epochs=param['dec_epochs'], path="./plots/{}/".format(param['name']))

    # test(model)