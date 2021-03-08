import itertools
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optimizer

from .dataset import TrainingData, TestingData
from .dnn import TripletNetwork, Encoder, DecoderR, DecoderD


class BuildMAT2(nn.Module):
    def __init__(
            self,
            data: pd.Series,
            metadata: pd.Series,
            mode: str = 'supervised',
            latent_num: int = 20,
            dropout_rate: float = 0.0,
            learning_rate: float = 5e-4,
            num_workers: int = 2,
            batch_size: int = 256,
            use_gpu: bool = True,
            anchor: pd.Series = None,
            norm: str = 'l1',
            mix_rate: float = 1.0,
            weight_decay: float = 0.01):
        super().__init__()

        # check input
        mode_list = ['supervised', 'semi-supervised', 'manual']
        if mode not in mode_list:
            print('Please specify a correct mode for triplet building!')
            raise IOError
        if 'batch' not in metadata.columns:
            print('Please check whether there is a \'batch\' column in metadata!')
            raise IOError
        if 'cluster' not in metadata.columns:
            metadata['cluster'] = metadata['batch']
        if anchor is not None:
            p_col = anchor.columns
            if anchor.shape[0] == 0 or \
                    'cell1' not in p_col or \
                    'cell2' not in p_col or \
                    'score' not in p_col:
                print('Please specify a correct file for anchors!')
                raise IOError
        if mode == 'supervised' or mode == 'semi-supervised':
            if 'type' not in metadata.columns:
                print('Please check whether there is a \'type\' column in metadata!')
                raise IOError

        self.mode = mode

        # set torch.device: cpu / single gpu
        self.device = torch.device("cpu")
        if use_gpu:
            if torch.cuda.is_available():
                self.device = torch.device("cuda")
            else:
                use_gpu = False

        # define training dataset
        self.data = TrainingData(
            data=data,
            metadata=metadata,
            mode=mode,
            anchor=anchor,
            mix_rate=mix_rate,
            shuffle=True,
            num_workers=num_workers,
            batch_size=batch_size,
            pin_memory=use_gpu,
            norm=norm)

        # set network
        gene_num = len(data)
        self._one_hot_num = len(set(metadata['batch']))
        self.encoder = Encoder(
            gene_num=gene_num,
            latent_num=latent_num,
            dropout_rate=dropout_rate)
        self.decoder_r = DecoderR(
            gene_num=gene_num,
            latent_num=latent_num)
        self.decoder_d = DecoderD(
            gene_num=gene_num,
            latent_num=latent_num,
            one_hot_num=self._one_hot_num)
        self._en_triplet_net = TripletNetwork(subnet=self.encoder)
        self._de_d_triplet_net = TripletNetwork(subnet=self.decoder_d)
        self._de_r_triplet_net = TripletNetwork(subnet=self.decoder_r)
        self.set_device(self.device)

        # set optimizer
        self._first_stage_optimizer = optimizer.Adam(
            self.encoder.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay)

        self._second_stage_optimizer = optimizer.Adam(
            itertools.chain(
                self.decoder_d.parameters(),
                self.decoder_r.parameters()),
            lr=learning_rate,
            weight_decay=weight_decay)

        # triplet record
        self.triplet_record = []

    def forward(self, cell):
        output = self.decoder_r(self.encoder(cell))
        return output

    def set_device(self, device: torch.device = "cpu"):
        self.device = device
        self.encoder = self.encoder.to(device)
        self.decoder_r = self.decoder_r.to(device)
        self.decoder_d = self.decoder_d.to(device)

    def train(self, epochs: int = 30, record: bool = False):
        self.triplet_record = []
        epoch_start_time = time.time()
        for epoch_i in range(epochs):
            train_loss = self.one_epoch_encoder(
                record_available=record)
            progress = ('#' * int(float(epoch_i + 1) /
                                  epochs * 30 + 0.5)).ljust(30)
            print(
                'Stage 1: [ %03d / %03d ] %6.2f sec(s) | %s | Train Loss: %3.6f' %
                (epoch_i + 1, epochs, (time.time() - epoch_start_time),
                    progress, train_loss), end='\r', flush=True)
        print("\n")
        epoch_start_time = time.time()
        for epoch_i in range(epochs):
            train_loss = self.one_epoch_decoder()
            progress = ('#' * int(float(epoch_i + 1) /
                                  epochs * 30 + 0.5)).ljust(30)
            print(
                'Stage 2: [ %03d / %03d ] %6.2f sec(s) | %s | Train Loss: %3.6f' %
                (epoch_i + 1, epochs, (time.time() - epoch_start_time),
                    progress, train_loss), end='\r', flush=True)
        print("\nTraining finish!\n")
        if record:
            return self.triplet_record

    def one_epoch_encoder(self, record_available: bool = False):
        train_loss = None
        for data in self.data.loader(stage='encoder', record=record_available):
            if record_available:
                cell, pos_cell, neg_cell, \
                    pos_score, neg_score, \
                    index, pos_index, neg_index = data
                self.triplet_record.append(
                    np.vstack(
                        [index.numpy(), pos_index.numpy(), neg_index.numpy()]))
            else:
                cell, pos_cell, neg_cell, \
                    pos_score, neg_score = data
            train_loss = self.encoder_loss(
                cell,
                pos_cell,
                neg_cell,
                pos_score,
                neg_score)

            self._first_stage_optimizer.zero_grad()
            train_loss.backward()
            self._first_stage_optimizer.step()
        return train_loss.detach()

    def one_epoch_decoder(self):
        train_loss = None
        one_hot = torch.eye(self._one_hot_num).requires_grad_(True)
        labels = self.data.get_attr(attr="label")
        for data in self.data.loader(stage='decoder', record=True):
            cell, pos_cell, neg_cell, _, _, \
                index, pos_index, neg_index = data
            train_loss = self.decoder_loss(cell,
                                           pos_cell,
                                           neg_cell,
                                           one_hot[labels[index], ],
                                           one_hot[labels[pos_index], ],
                                           one_hot[labels[neg_index], ])
            self._second_stage_optimizer.zero_grad()
            train_loss.backward()
            self._second_stage_optimizer.step()
        return train_loss.detach()

    def encoder_loss(
            self,
            cell,
            pos_cell,
            neg_cell,
            pos_score,
            neg_score):
        logic1 = pos_score > 1e-5
        logic2 = neg_score > 1e-5
        logic = logic1 | logic2
        zero = torch.Tensor([0.0]).to(self.device)
        pos_score = pos_score[logic].to(
            torch.float32).to(self.device)
        neg_score = neg_score[logic].to(
            torch.float32).to(self.device)

        cell, pos_cell, neg_cell = \
            cell[logic].to(torch.float32).to(self.device), \
            pos_cell[logic].to(torch.float32).to(self.device), \
            neg_cell[logic].to(torch.float32).to(self.device)

        dist_pos, dist_neg, _, _, _ = \
            self._en_triplet_net(cell, pos_cell, neg_cell)
        train_loss = torch.mean(
            torch.max(dist_pos - dist_neg +
                      (pos_score + neg_score) / 2, zero))
        return train_loss

    def decoder_loss(
            self,
            ori_cell,
            ori_pos_cell,
            ori_neg_cell,
            one_hot_cell,
            one_hot_pos_cell,
            one_hot_neg_cell):
        zero = torch.Tensor([0.0]).to(self.device)
        alpha = torch.Tensor([1.0]).to(self.device)
        ori_cell, ori_pos_cell, ori_neg_cell = \
            ori_cell.to(torch.float32).to(self.device), \
            ori_pos_cell.to(torch.float32).to(self.device), \
            ori_neg_cell.to(torch.float32).to(self.device)
        one_hot_cell, one_hot_pos_cell, one_hot_neg_cell = \
            one_hot_cell.to(torch.float32).to(self.device), \
            one_hot_pos_cell.to(torch.float32).to(self.device), \
            one_hot_neg_cell.to(torch.float32).to(self.device)

        _, _, en_ori_cell, en_ori_pos_cell, en_ori_neg_cell = \
            self._en_triplet_net(ori_cell, ori_pos_cell, ori_neg_cell)

        one_hot_cell = torch.cat(
            (en_ori_cell, one_hot_cell), 1)
        one_hot_pos_cell = torch.cat(
            (en_ori_pos_cell, one_hot_pos_cell), 1)
        one_hot_neg_cell = torch.cat(
            (en_ori_neg_cell, one_hot_neg_cell), 1)

        mse_loss = nn.MSELoss()

        _, _, rec_r_cell, rec_r_pos_cell, rec_r_neg_cell = \
            self._de_r_triplet_net(en_ori_cell,
                                   en_ori_pos_cell,
                                   en_ori_neg_cell)
        dist_pos, dist_neg, rec_d_cell, rec_d_pos_cell, rec_d_neg_cell = \
            self._de_d_triplet_net(one_hot_cell,
                                   one_hot_pos_cell,
                                   one_hot_neg_cell)
        recon, pos_recon, neg_recon = rec_r_cell + rec_d_cell, \
            rec_r_pos_cell + rec_d_pos_cell, rec_r_neg_cell + rec_d_neg_cell

        train_loss = torch.mean(
            torch.max(dist_pos - dist_neg + alpha, zero)) + \
            (mse_loss(recon, ori_cell) +
             mse_loss(pos_recon, ori_pos_cell) +
             mse_loss(neg_recon, ori_neg_cell)) / 3
        return train_loss

    def evaluate(self, data: pd.Series):
        dataloader = TestingData(data=data, model=self).loader
        rec = torch.Tensor(0).to(self.device)
        for cell, index in dataloader:
            cell = cell.to(torch.float32).to(self.device)
            output = self.forward(cell)
            rec = torch.cat([rec, output], 0)
        return rec.cpu().detach().numpy()
