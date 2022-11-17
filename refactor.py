# -*- coding: utf-8 -*-
import math
import os
import random
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import iqr, rankdata
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
from torch.nn import Linear, Parameter
from torch.utils.data import DataLoader, Subset
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.inits import glorot, zeros
from torch_geometric.utils import add_self_loops, remove_self_loops, softmax

from datasets.TimeDataset import TimeDataset


class GraphLayer(MessagePassing):
    """
    3.5 Graph attention based forecasting

    Feature extractos: to capture the relationships between sensors, we
    introduce a graph attention-based feature extractor to fuse a node's
    information with its neighbours based on the learned graph structure.

    """

    def __init__(
        self,
        in_channels,
        out_channels,
        heads=1,
        concat=True,
        negative_slope=0.2,
        dropout=0,
        bias=True,
    ):
        super(GraphLayer, self).__init__(aggr="add")

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.concat = concat
        self.negative_slope = negative_slope
        self.dropout = dropout

        self.__alpha__ = None

        self.lin = Linear(in_channels, heads * out_channels, bias=False)

        self.att_i = Parameter(torch.Tensor(1, heads, out_channels))
        self.att_j = Parameter(torch.Tensor(1, heads, out_channels))
        self.att_em_i = Parameter(torch.Tensor(1, heads, out_channels))
        self.att_em_j = Parameter(torch.Tensor(1, heads, out_channels))

        if bias and concat:
            self.bias = Parameter(torch.Tensor(heads * out_channels))
        elif bias and not concat:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter("bias", None)

        self.reset_parameters()

    def reset_parameters(self):
        glorot(self.lin.weight)
        glorot(self.att_i)
        glorot(self.att_j)
        zeros(self.att_em_i)
        zeros(self.att_em_j)
        zeros(self.bias)

    def forward(self, x, edge_index, embedding, return_attention_weights=False):

        if torch.is_tensor(x):
            x = self.lin(x)
            x = (x, x)
        else:
            x = (self.lin(x[0]), self.lin(x[1]))

        edge_index, _ = remove_self_loops(edge_index)
        edge_index, _ = add_self_loops(edge_index, num_nodes=x[1].size(self.node_dim))

        out = self.propagate(
            edge_index,
            x=x,
            embedding=embedding,
            edges=edge_index,
            return_attention_weights=return_attention_weights,
        )

        if self.concat:
            out = out.view(-1, self.heads * self.out_channels)
        else:
            out = out.mean(dim=1)

        if self.bias is not None:
            out = out + self.bias

        if return_attention_weights:
            alpha, self.__alpha__ = self.__alpha__, None
            return out, (edge_index, alpha)
        else:
            return out

    def message(
        self, x_i, x_j, edge_index_i, size_i, embedding, edges, return_attention_weights
    ):

        x_i = x_i.view(-1, self.heads, self.out_channels)
        x_j = x_j.view(-1, self.heads, self.out_channels)

        if embedding is not None:
            embedding_i, embedding_j = embedding[edge_index_i], embedding[edges[0]]
            embedding_i = embedding_i.unsqueeze(1).repeat(1, self.heads, 1)
            embedding_j = embedding_j.unsqueeze(1).repeat(1, self.heads, 1)

            key_i = torch.cat(
                (x_i, embedding_i), dim=-1
            )  # key_i's are the g_i's, does x_i already have W?
            key_j = torch.cat(
                (x_j, embedding_j), dim=-1
            )  # concatenates along the last dim, i.e. columns in this case

        cat_att_i = torch.cat((self.att_i, self.att_em_i), dim=-1)
        cat_att_j = torch.cat((self.att_j, self.att_em_j), dim=-1)

        alpha = (key_i * cat_att_i).sum(-1) + (key_j * cat_att_j).sum(
            -1
        )  # eqn (6) but...
        alpha = alpha.view(-1, self.heads, 1)
        alpha = F.leaky_relu(alpha, self.negative_slope)  # eqn (7)
        alpha = softmax(alpha, edge_index_i, size_i)  # eqn (8)

        if return_attention_weights:
            self.__alpha__ = alpha

        alpha = F.dropout(alpha, p=self.dropout, training=self.training)

        return x_j * alpha.view(-1, self.heads, 1)

    def __repr__(self):
        return "{}({}, {}, heads={})".format(
            self.__class__.__name__, self.in_channels, self.out_channels, self.heads
        )


class OutLayer(nn.Module):
    """
    Output layer, elementwise multiply representations, z_i, with the corresponding
    timeseries embedding, v_i, and use the
    """

    # in_num=64, layer_num=1, inter_num=256
    def __init__(self, in_num, layer_num, inter_num=512):

        super(OutLayer, self).__init__()
        modules = []
        for i in range(layer_num):
            if i == layer_num - 1:
                modules.append(nn.Linear(in_num if layer_num == 1 else inter_num, 1))
            else:
                layer_in_num = in_num if i == 0 else inter_num
                modules.append(nn.Linear(layer_in_num, inter_num))
                modules.append(nn.BatchNorm1d(inter_num))
                modules.append(nn.ReLU())

        self.mlp = nn.ModuleList(modules)

    def forward(self, x):
        out = x

        for mod in self.mlp:
            if isinstance(mod, nn.BatchNorm1d):
                out = out.permute(0, 2, 1)
                out = mod(out)
                out = out.permute(0, 2, 1)
            else:
                out = mod(out)

        return out


class GNNLayer(nn.Module):
    """
    Returning z_i = ReLU(...), the representations for all N nodes
    """

    def __init__(self, in_channel, out_channel, heads=1):
        super(GNNLayer, self).__init__()

        self.gnn = GraphLayer(in_channel, out_channel, heads=heads, concat=False)

        self.bn = nn.BatchNorm1d(out_channel)
        self.relu = nn.ReLU()
        self.leaky_relu = nn.LeakyReLU()

    def forward(self, x, edge_index, embedding=None):

        out, (new_edge_index, att_weight) = self.gnn(
            x, edge_index, embedding, return_attention_weights=True
        )  # out, (edge_index, alpha)

        self.att_weight_1 = att_weight
        self.edge_index_1 = new_edge_index

        out = self.bn(out)

        return self.relu(out)


class GDN(nn.Module):
    def __init__(
        self,
        fc_edge_idx,
        n_nodes,
        embed_dim=64,
        out_layer_inter_dim=256,
        input_dim=10,
        out_layer_num=1,
        topk=20,
    ):
        super(GDN, self).__init__()

        self.fc_edge_idx = fc_edge_idx
        self.n_nodes = n_nodes
        self.embed_dim = embed_dim
        self.out_layer_inter_dim = out_layer_inter_dim
        self.input_dim = input_dim
        self.out_layer_num = out_layer_num
        self.topk = topk

    def _initialise_layers(self):

        self.embedding = nn.Embedding(self.n_nodes, self.embed_dim)
        self.bn_outlayer_in = nn.BatchNorm1d(self.embed_dim)

        self.gnn_layers = nn.ModuleList(
            [
                GNNLayer(
                    self.input_dim,
                    self.embed_dim,
                    heads=1,
                )
            ]
        )

        self.node_embedding = None
        self.learned_graph = None

        self.out_layer = OutLayer(
            self.embed_dim, self.out_layer_num, inter_num=self.out_layer_inter_dim
        )

        self.cache_fc_edge_idx = None
        self.cache_embed_index = None

        self.dp = nn.Dropout(0.2)
        nn.init.kaiming_uniform_(self.embedding.weight, a=math.sqrt(5))

    def forward(self, data, org_edge_index=None):  # FIXME

        x = data.clone().detach()
        device = data.device
        batch_num, n_nodes, all_feature = x.shape
        x = x.view(-1, all_feature).contiguous()

        if self.cache_fc_edge_idx is None:
            self.cache_fc_edge_idx = get_batch_edge_index(
                self.fc_edge_idx, batch_num, n_nodes
            ).to(device)

        all_embeddings = self.embedding(torch.arange(n_nodes).to(device))  # v_i's

        weights_arr = all_embeddings.detach().clone()
        all_embeddings = all_embeddings.repeat(batch_num, 1)

        weights = weights_arr.view(n_nodes, -1)

        cos_ji_mat = torch.matmul(weights, weights.T)  # e_{ji} in eqn (2)
        normed_mat = torch.matmul(
            weights.norm(dim=-1).view(-1, 1), weights.norm(dim=-1).view(1, -1)
        )
        cos_ji_mat = cos_ji_mat / normed_mat

        topk_indices_ji = torch.topk(cos_ji_mat, self.topk, dim=-1)[
            1
        ]  # A_{ji} in eqn (3)

        self.learned_graph = topk_indices_ji

        gated_i_ = torch.arange(0, n_nodes)
        gated_i = (
            gated_i_.permute(*torch.arange(gated_i_.ndim - 1, -1, -1))
            .unsqueeze(1)
            .repeat(1, self.topk)
            .flatten()
            .to(device)
            .unsqueeze(0)
        )

        gated_j = topk_indices_ji.flatten().unsqueeze(0)
        gated_edge_index = torch.cat((gated_j, gated_i), dim=0)

        batch_gated_edge_index = get_batch_edge_index(
            gated_edge_index, batch_num, n_nodes
        ).to(device)

        gcn_out = self.gnn_layers[0](
            x,
            batch_gated_edge_index,
            embedding=all_embeddings,
        )
        gcn_out = gcn_out.view(batch_num, n_nodes, -1)

        idxs = torch.arange(0, n_nodes).to(device)
        out = torch.mul(gcn_out, self.embedding(idxs))
        out = out.permute(0, 2, 1)
        out = F.relu(self.bn_outlayer_in(out))
        out = out.permute(0, 2, 1)
        out = self.dp(out)
        out = self.out_layer(out)
        out = out.view(-1, n_nodes)

        return out


def get_batch_edge_index(org_edge_index, batch_num, n_nodes):

    edge_index = org_edge_index.clone().detach()
    edge_num = org_edge_index.shape[1]
    batch_edge_index = edge_index.repeat(1, batch_num).contiguous()

    for i in range(batch_num):
        batch_edge_index[:, i * edge_num : (i + 1) * edge_num] += i * n_nodes

    return batch_edge_index.long()


class GNNAD:
    """
    Graph Neural Network-based Anomaly Detection in Multivariate Timeseries.
    """

    def __init__(
        self,
        batch: int = 128,
        epoch: int = 100,
        slide_win: int = 15,
        dim: int = 64,
        slide_stride: int = 5,
        comment: str = "",
        random_seed: int = 0,
        out_layer_num: int = 1,
        out_layer_inter_dim: int = 256,
        decay: float = 0,
        validate_ratio: float = 0.1,
        topk: int = 20,
        data_subdir: str = "msl",
        device: str = "cpu",
        report: str = "best",
        load_model_name: str = "",
        early_stop_win: int = 15,
        lr: float = 0.001,
    ):

        self.batch = batch
        self.epoch = epoch
        self.slide_win = slide_win
        self.dim = dim
        self.slide_stride = slide_stride
        self.comment = comment
        self.random_seed = random_seed
        self.out_layer_num = out_layer_num
        self.out_layer_inter_dim = out_layer_inter_dim
        self.decay = decay
        self.validate_ratio = validate_ratio
        self.topk = topk
        self.data_subdir = data_subdir
        self.device = device
        self.report = report
        self.load_model_name = load_model_name
        self.early_stop_win = early_stop_win
        self.lr = lr

    def _set_seeds(self):
        random.seed(self.random_seed)
        np.random.seed(self.random_seed)
        torch.manual_seed(self.random_seed)

    def _split_train_validation(self, data):

        dataset_len = len(data)
        validate_use_len = int(dataset_len * self.validate_ratio)
        validate_start_idx = random.randrange(dataset_len - validate_use_len)
        idx = torch.arange(dataset_len)

        train_sub_idx = torch.cat(
            [idx[:validate_start_idx], idx[validate_start_idx + validate_use_len :]]
        )
        train_subset = Subset(data, train_sub_idx)

        validate_sub_idx = idx[
            validate_start_idx : validate_start_idx + validate_use_len
        ]
        validate_subset = Subset(data, validate_sub_idx)

        return train_subset, validate_subset

    def _load_data(self):

        train = pd.read_csv(
            f"./data/{self.data_subdir}/train.csv", sep=",", index_col=0
        )
        test = pd.read_csv(f"./data/{self.data_subdir}/test.csv", sep=",", index_col=0)

        train = train.drop(columns=["attack"]) if "attack" in train.columns else train

        feature_list = train.columns[
            train.columns.str[0] != "_"
        ].to_list()  # convention is to pass non-features as '_'
        assert len(feature_list) == len(set(feature_list))

        fc_struc = {
            ft: [x for x in feature_list if x != ft] for ft in feature_list
        }  # fully connected structure

        edge__idx_tuples = [
            (feature_list.index(child), feature_list.index(node_name))
            for node_name, node_list in fc_struc.items()
            for child in node_list
        ]

        fc_edge_idx = [
            [x[0] for x in edge__idx_tuples],
            [x[1] for x in edge__idx_tuples],
        ]
        fc_edge_idx = torch.tensor(fc_edge_idx, dtype=torch.long)

        train_input = parse_data(train, feature_list)
        test_input = parse_data(test, feature_list, labels=test.attack.tolist())

        cfg = {
            "slide_win": self.slide_win,
            "slide_stride": self.slide_stride,
        }

        train_dataset = TimeDataset(train_input, fc_edge_idx, mode="train", config=cfg)
        test_dataset = TimeDataset(test_input, fc_edge_idx, mode="test", config=cfg)

        train_subset, validate_subset = self._split_train_validation(train_dataset)

        # get data loaders
        train_dataloader = DataLoader(
            train_subset, batch_size=self.batch, shuffle=False, num_workers=0
        )  # FIXME: shuffle=True

        validate_dataloader = DataLoader(
            validate_subset,
            batch_size=self.batch,
            shuffle=False,
            num_workers=0,  # FIXME: num_workers=0
        )

        test_dataloader = DataLoader(
            test_dataset,
            batch_size=self.batch,
            shuffle=False,
            num_workers=0,  # FIXME: num_workers=0
        )

        # save to self
        self.fc_edge_idx = fc_edge_idx
        self.feature_list = feature_list
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.train_dataloader = train_dataloader
        self.validate_dataloader = validate_dataloader
        self.test_dataloader = test_dataloader

    def _load_model(self):
        # instantiate model
        model = GDN(
            self.fc_edge_idx,
            n_nodes=len(self.feature_list),
            input_dim=self.slide_win,
            out_layer_num=self.out_layer_num,
            out_layer_inter_dim=self.out_layer_inter_dim,
            topk=self.topk,
        ).to(self.device)

        model._initialise_layers()

        self.model = model

    def _get_model_path(self):
        # f'./results/{self.data_subdir}/{model_name}.csv'

        datestr = datetime.now().strftime("%m%d-%H%M%S")
        model_name = datestr if len(self.load_model_name) == 0 else self.load_model_name
        model_path = f"./pretrained/{self.data_subdir}/{model_name}.pt"
        dirname = os.path.dirname(model_path)
        Path(dirname).mkdir(parents=True, exist_ok=True)

        self.model_path = model_path

    def _test(self, model, dataloader):

        start = datetime.now()

        test_loss_list = []
        acu_loss = 0

        t_test_predicted_list = []
        t_test_ground_list = []
        t_test_labels_list = []

        model.eval()

        for i, (x, y, labels, edge_index) in enumerate(dataloader):
            x, y, labels, edge_index = [
                item.to(self.device).float() for item in [x, y, labels, edge_index]
            ]

            with torch.no_grad():
                predicted = model(x).float().to(self.device)
                loss = loss_func(predicted, y)
                labels = labels.unsqueeze(1).repeat(1, predicted.shape[1])

                if len(t_test_predicted_list) <= 0:
                    t_test_predicted_list = predicted
                    t_test_ground_list = y
                    t_test_labels_list = labels
                else:
                    t_test_predicted_list = torch.cat(
                        (t_test_predicted_list, predicted), dim=0
                    )
                    t_test_ground_list = torch.cat((t_test_ground_list, y), dim=0)
                    t_test_labels_list = torch.cat((t_test_labels_list, labels), dim=0)

            test_loss_list.append(loss.item())
            acu_loss += loss.item()

            if i % 10000 == 1 and i > 1:
                print(str_time_elapsed(start, i, len(dataloader)))

        test_predicted_list = t_test_predicted_list.tolist()
        test_ground_list = t_test_ground_list.tolist()
        test_labels_list = t_test_labels_list.tolist()

        avg_loss = sum(test_loss_list) / len(test_loss_list)

        return avg_loss, [test_predicted_list, test_ground_list, test_labels_list]

    def _train(self):

        optimizer = torch.optim.Adam(
            self.model.parameters(), lr=self.lr, weight_decay=self.decay
        )

        train_log = []
        max_loss = 1e8
        stop_improve_count = 0

        for i_epoch in range(self.epoch):

            acu_loss = 0
            self.model.train()

            for i, (x, labels, _, edge_index) in enumerate(self.train_dataloader):

                x, labels, edge_index = [
                    item.float().to(self.device) for item in [x, labels, edge_index]
                ]

                optimizer.zero_grad()

                out = self.model(x).float().to(self.device)

                loss = loss_func(out, labels)

                loss.backward()
                optimizer.step()

                train_log.append(loss.item())
                acu_loss += loss.item()

            # each epoch
            print(
                "epoch ({} / {}) (Loss:{:.8f}, ACU_loss:{:.8f})".format(
                    i_epoch, self.epoch, acu_loss / (i + 1), acu_loss
                ),
                flush=True,
            )

            # use val dataset to judge
            if self.validate_dataloader is not None:

                val_loss, _ = self._test(self.model, self.validate_dataloader)

                if val_loss < max_loss:
                    torch.save(self.model.state_dict(), self.model_path)

                    max_loss = val_loss
                    stop_improve_count = 0
                else:
                    stop_improve_count += 1

                if stop_improve_count >= self.early_stop_win:
                    break

            else:
                if acu_loss < max_loss:
                    torch.save(self.model.state_dict(), self.model_path)
                    max_loss = acu_loss

        self.train_log = train_log

    def _get_score(self):

        # read in best model
        self.model.load_state_dict(torch.load(self.model_path))
        best_model = self.model.to(self.device)

        # store results to self
        _, self.test_result = self._test(best_model, self.test_dataloader)
        _, self.validate_result = self._test(best_model, self.validate_dataloader)

        test_result = np.array(self.test_result)
        test_labels = test_result[2, :, 0].tolist()
        test_scores = get_full_err_scores(test_result)

        info = get_best_performance_data(test_scores, test_labels, topk=1)

        print("=========================** Result **============================\n")
        print(f"F1 score: {info[0]}")
        print(f"precision: {info[1]}")
        print(f"recall: {info[2]}\n")

    def fit(self):
        self._set_seeds()
        self._load_data()
        self._load_model()
        self._get_model_path()
        self._train()
        self._get_score()

        return self


def loss_func(y_pred, y_true):
    return F.mse_loss(y_pred, y_true, reduction="mean")


def parse_data(data, feature_list, labels=None):

    labels = [0] * data.shape[0] if labels == None else labels
    res = data[feature_list].T.values.tolist()
    res.append(labels)
    return res


def str_seconds_to_minutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return "%dm %ds" % (m, s)


def str_time_elapsed(start, i, total):
    now = datetime.now()
    elapsed = (now - start).seconds
    frac_complete = (i + 1) / total
    remaining = elapsed / frac_complete - elapsed
    return "%s (- %s)" % (
        str_seconds_to_minutes(elapsed),
        str_seconds_to_minutes(remaining),
    )


def get_full_err_scores(test_result):
    test_result = np.array(test_result)

    all_scores = None
    feature_num = test_result.shape[-1]

    for i in range(feature_num):
        test_result_list = test_result[:2, :, i]
        scores = get_err_scores(test_result_list)

        if all_scores is None:
            all_scores = scores
        else:
            all_scores = np.vstack((all_scores, scores))

    return all_scores


def get_err_scores(test_result_list):
    test_predict, test_ground = test_result_list

    n_err_mid, n_err_iqr = get_err_median_and_iqr(test_predict, test_ground)

    test_delta = np.abs(
        np.subtract(
            np.array(test_predict).astype(np.float64),
            np.array(test_ground).astype(np.float64),
        )
    )
    epsilon = 1e-2

    err_scores = (test_delta - n_err_mid) / (np.abs(n_err_iqr) + epsilon)

    smoothed_err_scores = np.zeros(err_scores.shape)
    before_num = 3
    for i in range(before_num, len(err_scores)):
        smoothed_err_scores[i] = np.mean(err_scores[i - before_num : i + 1])

    return smoothed_err_scores


def get_err_median_and_iqr(predicted, groundtruth):

    np_arr = np.abs(np.subtract(np.array(predicted), np.array(groundtruth)))

    err_median = np.median(np_arr)
    err_iqr = iqr(np_arr)

    return err_median, err_iqr


def get_best_performance_data(total_err_scores, gt_labels, topk=1):

    total_features = total_err_scores.shape[0]

    topk_indices = np.argpartition(
        total_err_scores, range(total_features - topk - 1, total_features), axis=0
    )[-topk:]

    total_topk_err_scores = []

    total_topk_err_scores = np.sum(
        np.take_along_axis(total_err_scores, topk_indices, axis=0), axis=0
    )

    final_topk_fmeas, thresolds = eval_scores(
        total_topk_err_scores, gt_labels, 400, return_thresold=True
    )

    th_i = final_topk_fmeas.index(max(final_topk_fmeas))
    thresold = thresolds[th_i]

    pred_labels = np.zeros(len(total_topk_err_scores))
    pred_labels[total_topk_err_scores > thresold] = 1

    for i in range(len(pred_labels)):
        pred_labels[i] = int(pred_labels[i])
        gt_labels[i] = int(gt_labels[i])

    pre = precision_score(gt_labels, pred_labels)
    rec = recall_score(gt_labels, pred_labels)

    auc_score = roc_auc_score(gt_labels, total_topk_err_scores)

    return max(final_topk_fmeas), pre, rec, auc_score, thresold


# calculate F1 scores
def eval_scores(scores, true_scores, th_steps, return_thresold=False):
    padding_list = [0] * (len(true_scores) - len(scores))

    if len(padding_list) > 0:
        scores = padding_list + scores

    scores_sorted = rankdata(scores, method="ordinal")
    th_vals = np.array(range(th_steps)) * 1.0 / th_steps
    fmeas = [None] * th_steps
    thresholds = [None] * th_steps

    for i in range(th_steps):
        cur_pred = scores_sorted > th_vals[i] * len(scores)

        fmeas[i] = f1_score(true_scores, cur_pred)

        score_index = scores_sorted.tolist().index(int(th_vals[i] * len(scores) + 1))
        thresholds[i] = scores[score_index]

    if return_thresold:
        return fmeas, thresholds
    return fmeas
