import torch
from torch import nn
from torch.nn import Parameter


class AutoEncoder(nn.Module):
	def __init__(self, hidden, input_size):
		super(AutoEncoder, self).__init__()
		self.encoder = nn.Sequential(
			nn.Linear(input_size, 500),
			nn.ReLU(True),
			nn.Linear(500, 500),
			nn.ReLU(True),
			nn.Linear(500, 500),
			nn.ReLU(True),
			nn.Linear(500, 2000),
			nn.ReLU(True),
			nn.Linear(2000, hidden))
		self.decoder = nn.Sequential(
			nn.Linear(hidden, 2000),
			nn.ReLU(True),
			nn.Linear(2000, 500),
			nn.ReLU(True),
			nn.Linear(500, 500),
			nn.ReLU(True),
			nn.Linear(500, 500),
			nn.ReLU(True),
			nn.Linear(500, input_size))
		self.model = nn.Sequential(self.encoder, self.decoder)

	def encode(self, x):
		return self.encoder(x)

	def decode(self, x):
		return self.decoder(x)

	def forward(self, x):
		return self.model(x)


class GCML(nn.Module):

	def __init__(self, n_clusters, hidden, autoencoder, alpha=1.0):
		super(GCML, self).__init__()
		self.n_clusters = n_clusters
		self.alpha = alpha
		self.autoencoder = autoencoder

		self.cluster_centers = Parameter(torch.Tensor(n_clusters, hidden))
		torch.nn.init.xavier_normal_(self.cluster_centers)

	def target_distribution(self, q):
		weight = (q ** 2.0) / torch.sum(q, 0)
		return (weight.t() / torch.sum(weight, 1)).t()

	def forward(self, x):

		hidden = self.autoencoder.encode(x)
		x_rec = self.autoencoder.decode(hidden)

		p_squared = torch.sum((hidden.unsqueeze(1) - self.cluster_centers)**2, 2)
		p = 1.0 / (1.0 + (p_squared / self.alpha))
		power = float(self.alpha + 1) / 2
		p = p ** power
		p_dist = (p.t() / torch.sum(p, 1)).t()

		return x_rec, p_dist

