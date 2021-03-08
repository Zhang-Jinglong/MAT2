from collections import defaultdict

import numpy as np
import pandas as pd
from torch.utils.data import Dataset


def get_encoder_anchor_list(metadata: pd.Series):
    annotation = metadata
    annotation_group = annotation.dropna().groupby(["batch", "type"]).groups
    annotation_group_keys = list(annotation_group.keys())

    positive_anchor = defaultdict(list)
    negative_anchor = defaultdict(list)

    for batch, cell_type in annotation_group_keys:
        batch_mismatch = np.where(annotation['batch'] != batch)[0]
        type_match = np.where(annotation['type'] == cell_type)[0]
        type_mismatch = np.where(annotation['type'] != cell_type)[0]
        positive_match = np.intersect1d(batch_mismatch, type_match)
        if len(positive_match) > 0:
            positive_anchor[(batch, cell_type)].append(positive_match)
        else:
            positive_anchor[(batch, cell_type)].append(type_match)
        negative_match = np.intersect1d(type_mismatch,
                                        np.where(~annotation.isna().any(1)))
        negative_anchor[(batch, cell_type)].append(negative_match)
    return positive_anchor, negative_anchor


def get_decoder_anchor_list(metadata: pd.Series):
    annotation = metadata[['batch', "cluster"]]
    annotation_group = annotation.groupby(["batch", "cluster"]).groups
    annotation_group_keys = list(annotation_group.keys())
    positive_anchor = defaultdict(list)
    negative_anchor = defaultdict(list)

    for batch, cluster in annotation_group_keys:
        batch_mismatch = np.where(annotation['batch'] != batch)[0]
        batch_match = np.where(annotation['batch'] == batch)[0]
        cluster_match = np.where(annotation['cluster'] != cluster)[0]
        positive_match = np.intersect1d(batch_match, cluster_match)
        if len(positive_match) > 0:
            positive_anchor[(batch, cluster)].append(positive_match)
        else:
            positive_anchor[(batch, cluster)].append(batch_match)
        negative_anchor[(batch, cluster)].append(batch_mismatch)
    return positive_anchor, negative_anchor


class EncoderSupervisedTriplet:
    def __init__(self, normalized_data):
        self._data = normalized_data.dataset
        self._metadata = normalized_data.metadata
        pos_anchor, neg_anchor = get_encoder_anchor_list(
            self._metadata[['batch', "type"]])
        self.pos_anchor = pos_anchor
        self.neg_anchor = neg_anchor
        self.type_na = self._metadata.isna()['type']

    def triplet(self, index):
        index_batch = self._metadata["batch"][index]
        index_type = self._metadata["type"][index]
        pos_anchor = self.pos_anchor[(index_batch, index_type)][0]
        pos_num = len(pos_anchor)
        pos_index = pos_anchor[np.random.randint(pos_num)]
        pos_cell = self._data[pos_index]

        neg_anchor = self.neg_anchor[(index_batch, index_type)][0]
        neg_num = len(neg_anchor)
        neg_index = neg_anchor[np.random.randint(neg_num)]
        neg_cell = self._data[neg_index]
        cell = self._data[index]
        pos_score = 1.0
        neg_score = 1.0

        return dict(
            anchor=dict(cell=cell, index=index),
            positive=dict(cell=pos_cell, index=pos_index, score=pos_score),
            negative=dict(cell=neg_cell, index=neg_index, score=neg_score))


class EncoderManualTriplet:
    def __init__(
            self,
            normalized_data,
            anchor: pd.Series):
        self._data = normalized_data.dataset
        pos_anchor = defaultdict(list)
        neg_anchor = defaultdict(list)
        for item in range(len(anchor)):
            cell1 = int(anchor['cell1'].iloc[item])
            cell2 = int(anchor['cell2'].iloc[item])
            score = float(anchor['score'].iloc[item])
            if score < 0.0:
                neg_anchor[cell1].append([cell2, abs(score)])
            else:
                pos_anchor[cell1].append([cell2, score])

        self.pos_anchor = pos_anchor
        self.neg_anchor = neg_anchor

    def triplet(self, index):
        cell = self._data[index]
        pos_num = len(self.pos_anchor[index])
        if pos_num > 0:
            pos_index, pos_score = self.pos_anchor[index][np.random.randint(
                pos_num)]
            neg_num = len(self.neg_anchor[index])
            if neg_num > 0:
                neg_index, neg_score = self.neg_anchor[index][np.random.randint(
                    neg_num)]
            else:
                neg_index = np.random.randint(len(self._data))
                if neg_index != index and neg_index != pos_index:
                    neg_score = 1.0
                else:
                    neg_index = index
                    neg_score = -pos_score
            pos_cell = self._data[pos_index]
            neg_cell = self._data[neg_index]
        else:
            pos_index = index
            pos_score = 0.0
            neg_index = index
            neg_score = 0.0
            pos_cell = self._data[pos_index]
            neg_cell = self._data[neg_index]

        return dict(
            anchor=dict(cell=cell, index=index),
            positive=dict(cell=pos_cell, index=pos_index, score=pos_score),
            negative=dict(cell=neg_cell, index=neg_index, score=neg_score))


class EncoderTriplet:
    def __init__(
            self,
            normalized_data,
            anchor: pd.Series,
            mode: str,
            mix_rate: float):

        self.s_triplet, self.m_triplet = None, None
        if mode == "supervised" or mode == 'semi-supervised':
            self.s_triplet = EncoderSupervisedTriplet(
                normalized_data=normalized_data)
        if mode == 'manual' or mode == 'semi-supervised':
            self.m_triplet = EncoderManualTriplet(
                normalized_data=normalized_data,
                anchor=anchor)
        self._mode = mode
        self._mix_rate = mix_rate

    def triplet(self, index):
        if self._mode == "supervised":
            return self.s_triplet.triplet(index)
        elif self._mode == "manual":
            return self.m_triplet.triplet(index)
        elif self._mode == "semi-supervised":
            p_sup = 1.0 / (1.0 + self._mix_rate)
            if ~self.s_triplet.type_na[index] and \
                    np.random.random() <= p_sup:
                return self.s_triplet.triplet(index)
            return self.m_triplet.triplet(index)


class DecoderTriplet:
    def __init__(self, normalized_data):
        self._data = normalized_data.dataset
        self._metadata = normalized_data.metadata
        pos_anchor, neg_anchor = get_decoder_anchor_list(
            normalized_data.metadata)
        self.pos_anchor = pos_anchor
        self.neg_anchor = neg_anchor

    def triplet(self, index):
        index_batch = self._metadata["batch"][index]
        index_cluster = self._metadata["cluster"][index]
        pos_anchor = self.pos_anchor[(index_batch, index_cluster)][0]
        pos_num = len(pos_anchor)
        pos_index = pos_anchor[np.random.randint(pos_num)]
        pos_cell = self._data[pos_index]

        neg_anchor = self.neg_anchor[(index_batch, index_cluster)][0]
        neg_num = len(neg_anchor)
        neg_index = neg_anchor[np.random.randint(neg_num)]
        neg_cell = np.array(self._data[neg_index])
        cell = self._data[index]

        return dict(anchor=dict(cell=cell, index=index),
                    positive=dict(cell=pos_cell, index=pos_index, score=1.0),
                    negative=dict(cell=neg_cell, index=neg_index, score=1.0))


class Triplet(Dataset):
    def __init__(self,
                 normalized_data,
                 mode: str,
                 mix_rate: float,
                 anchor: pd.Series = None,
                 record: bool = False):
        self.cell_num = len(normalized_data.dataset)
        self.en_triplet = EncoderTriplet(normalized_data=normalized_data,
                                         anchor=anchor,
                                         mode=mode,
                                         mix_rate=mix_rate)
        self.de_triplet = DecoderTriplet(normalized_data)
        self._stage = 'encoder'
        self._record = record

    def set_stage(self, stage: str = 'encoder'):
        self._stage = stage

    def set_record(self, record: bool = False):
        self._record = record

    def __getitem__(self, index):
        triplet = None
        if self._stage == "encoder":
            triplet = self.en_triplet.triplet(index)
        elif self._stage == 'decoder':
            triplet = self.de_triplet.triplet(index)
        if self._record:
            record = (
                triplet["anchor"]["cell"],
                triplet["positive"]["cell"],
                triplet["negative"]["cell"],
                triplet["positive"]["score"],
                triplet["negative"]["score"],
                triplet["anchor"]["index"],
                triplet["positive"]["index"],
                triplet["negative"]["index"])
        else:
            record = (
                triplet["anchor"]["cell"],
                triplet["positive"]["cell"],
                triplet["negative"]["cell"],
                triplet["positive"]["score"],
                triplet["negative"]["score"])
        return record

    def __len__(self):
        return self.cell_num
