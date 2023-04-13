import torch
import torch.nn as nn
import torch.nn.functional as F

class SoftKMeans(nn.Module):
    def __init__(self, n_clusters, seq_len,n_features, dist_metric='eucl', alpha=1.0):
        super(SoftKMeans, self).__init__()
        self.n_clusters = n_clusters
        self.n_features = n_features
        self.dist_metric = dist_metric
        self.alpha = alpha
        self.clusters = nn.Parameter(torch.randn(n_clusters, seq_len ,n_features))
        
    def forward_clusters(self, inputs):
        """
        Student t-distribution kernel, probability of assigning encoded sequence i to cluster k.
            q_{ik} = (1 + dist(z_i, m_k)^2)^{-1} / normalization.
        Arguments:
            inputs: encoded input sequences, shape=(n_samples, timesteps, n_features)
        Return:
            q: soft labels for each sample. shape=(n_samples, n_clusters)
        """ 
        if self.dist_metric == 'eucl':
            distance = (inputs.unsqueeze(1) - self.clusters).square().sum(dim=2).sqrt().sum(-1) # shape (n_samples,n_clusters)
        elif self.dist_metric == 'cid':
            ce_x = torch.sqrt(torch.sum(torch.square(inputs[:, 1:, :] - inputs[:, :-1, :]), dim=1))  # shape (n_samples, n_features)
            ce_w = torch.sqrt(torch.sum(torch.square(self.clusters[:, 1:, :] - self.clusters[:, :-1, :]), dim=1))  # shape (n_clusters, n_features)
            ce = torch.maximum(torch.unsqueeze(ce_x, dim=1), ce_w) / torch.minimum(torch.unsqueeze(ce_x, dim=1), ce_w)  # shape (n_samples, n_clusters, n_features)
            ed = torch.sqrt(torch.sum(torch.square(torch.unsqueeze(inputs, dim=1) - self.clusters), dim=2))  # shape (n_samples, n_clusters, n_features)
            distance = torch.sum(ed * ce, dim=-1)  # shape (n_samples, n_clusters)
        elif self.dist_metric == 'cor':
            inputs_norm = (inputs - torch.unsqueeze(torch.mean(inputs, dim=1), dim=1)) / torch.unsqueeze(torch.std(inputs, dim=1), dim=1)  # shape (n_samples, timesteps, n_features)
            clusters_norm = (self.clusters - torch.unsqueeze(torch.mean(self.clusters, dim=1), dim=1)) / torch.unsqueeze(torch.std(self.clusters, dim=1), dim=1)  # shape (n_clusters, timesteps, n_features)
            pcc = torch.mean(torch.unsqueeze(inputs_norm, dim=1) * clusters_norm, dim=2)  # Pearson correlation coefficients
            distance = torch.sum(torch.sqrt(2.0 * (1.0 - pcc)), dim=-1)  # correlation-based similarities, shape (n_samples, n_clusters)
        elif self.dist_metric == 'acf':
            raise NotImplementedError
        else:
            raise ValueError('Available distances are eucl, cid, cor and acf!')
        q = 1.0 / (1.0 + torch.square(distance) / self.alpha)
        q **= (self.alpha + 1.0) / 2.0
        q = torch.softmax(q,1)
        p = torch.softmax(q ** 2 / q.sum(0),1)
        return q,p

    def forward(self,inputs):
        q,p = self.forward_clusters(inputs)
        loss = nn.KLDivLoss(reduction="batchmean")(p,q)
        return q,loss