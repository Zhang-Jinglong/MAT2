import torch
import torch.nn as nn


def _distance(cell1, cell2, dist: str = 'norm2'):
    d = None
    if dist == 'norm2':
        d = nn.functional.pairwise_distance(cell1, cell2, p=2)
    elif dist == 'norm1':
        d = nn.functional.pairwise_distance(cell1, cell2, p=1)
    elif dist == 'cosine':
        d = 1.0 - torch.cosine_similarity(cell1, cell2, dim=1)
    return d


class TripletNetwork(nn.Module):
    def __init__(self, subnet, dist: str = 'norm2'):
        super().__init__()
        self.subnet = subnet
        self._dist = dist

    def forward(self, anchor, positive, negative):
        latent_anchor = self.subnet(anchor)
        latent_positive = self.subnet(positive)
        latent_negative = self.subnet(negative)
        dist_positive = _distance(
            latent_anchor, latent_positive, self._dist)
        dist_negative = _distance(
            latent_anchor, latent_negative, self._dist)
        return dist_positive, dist_negative, \
            latent_anchor, latent_positive, latent_negative

    def set_dist(self, dist: str = 'norm2'):
        self._dist = dist


class Encoder(nn.Module):
    def __init__(
            self,
            gene_num: int,
            latent_num: int = 20,
            dropout_rate: float = 0.3):
        super().__init__()
        self.layer1 = nn.Sequential(
            nn.Linear(gene_num, gene_num // 4),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate)
        )
        self.layer2 = nn.Sequential(
            nn.Linear(gene_num // 4, gene_num // 8),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate)
        )
        self.layer3 = nn.Sequential(
            nn.Linear(gene_num // 8, latent_num)
        )

    def forward(self, cell):
        output = self.layer3(self.layer2(self.layer1(cell)))
        return output


class DecoderR(nn.Module):
    def __init__(self, gene_num: int, latent_num: int = 20):
        super().__init__()
        self.layer1 = nn.Sequential(
            nn.Linear(latent_num, gene_num),
            nn.ReLU(inplace=True)
        )

    def forward(self, cell):
        output = self.layer1(cell)
        return output


class DecoderD(nn.Module):
    def __init__(
            self,
            gene_num: int,
            one_hot_num: int,
            latent_num: int = 20):
        super().__init__()
        self.layer1 = nn.Sequential(
            nn.Linear(latent_num + one_hot_num, gene_num // 8),
            nn.ReLU(inplace=True)
        )
        self.layer2 = nn.Sequential(
            nn.Linear(gene_num // 8, gene_num)
        )

    def forward(self, cell):
        output = self.layer2(self.layer1(cell))
        return output
