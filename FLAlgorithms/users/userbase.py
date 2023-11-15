# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0


import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import json
from torch.utils.data import DataLoader
import numpy as np
import copy

from utils.metrics import get_full_sort_score
from utils.sampler import WarpSampler

class User:
    """
    Base class for users in federated learning.
    """
    def __init__(self, device, usermodel_id, train_dict, train_spmat, train_graph, test_dict, itemnum, model, args):

        self.device = device
        self.model = copy.deepcopy(model).to(self.device)
        self.args = args
        self.usermodel_id = usermodel_id
        self.train_graph = train_graph.to(self.device)
        self.train_spmat = train_spmat
        self.train_dict = train_dict
        self.train_i2ipairs = self.train_spmat.nonzero()
        self.test_dict = test_dict
        self.itemnum = itemnum
        self.sampler = WarpSampler(self.train_i2ipairs, self.train_dict, self.itemnum, batch_size=self.args.batch_size, n_workers=8)
        self.batch_size = args.batch_size
        self.learning_rate = args.learning_rate
        self.beta = args.beta
        self.lamda = args.lamda
        self.local_epochs = args.local_epochs

        # those parameters are for personalized federated learing.
        self.local_model = copy.deepcopy(list(self.model.parameters()))
        self.personalized_model = copy.deepcopy(list(self.model.parameters()))
        self.personalized_model_bar = copy.deepcopy(list(self.model.parameters()))
    
    def set_parameters(self, model):
        for old_param, new_param, local_param in zip(self.model.parameters(), model.parameters(), self.local_model):
            old_param.data = new_param.data.clone().to(self.device)
            local_param.data = new_param.data.clone().to(self.device)
        #self.local_weight_updated = copy.deepcopy(self.optimizer.param_groups[0]['params'])

    def get_parameters(self):
        for param in self.model.parameters():
            param.detach()
        return self.model.parameters()
    
    def clone_model_paramenter(self, param, clone_param):
        for param, clone_param in zip(param, clone_param):
            clone_param.data = param.data.clone()
        return clone_param
    
    def get_updated_parameters(self):
        return self.local_weight_updated
    
    def update_parameters(self, new_params):
        for param , new_param in zip(self.model.parameters(), new_params):
            param.data = new_param.data.clone()

    def get_grads(self):
        grads = []
        for param in self.model.parameters():
            if param.grad is None:
                grads.append(torch.zeros_like(param.data))
            else:
                grads.append(param.grad.data)
        return grads


    def batch_predict(self, test_items):
        test_items = torch.tensor(np.array(test_items), dtype=torch.long).to(self.device)
        self.model = self.model.to(self.device)
        self.model.eval()
        item_embeddings = self.model(self.train_graph.x, self.train_graph.edge_index)

        preds = torch.matmul(item_embeddings[test_items], item_embeddings.transpose(0, 1))
        #self.model = self.model.to("cpu")
        return preds

    def test(self, eval_info, verbose=False):
        test_items = list(self.test_dict.keys())
        batch_test_items = []
        preds = None
        answers = []
        for i in range(1, len(test_items)+1):
            batch_test_items.append(test_items[i-1])
            if i % self.args.batch_size == 0:
                rating_pred = self.batch_predict(np.array(batch_test_items))
                rating_pred = rating_pred.cpu().data.numpy().copy()
                rating_pred[self.train_spmat[batch_test_items].toarray() > 0] = 0
                ind = np.argpartition(rating_pred, -40)[:, -40:]
                arr_ind = rating_pred[np.arange(len(rating_pred))[:, None], ind]
                arr_ind_argsort = np.argsort(arr_ind)[np.arange(len(rating_pred)), ::-1]
                batch_pred_list = ind[np.arange(len(rating_pred))[:, None], arr_ind_argsort]
                if preds is None:
                    preds = batch_pred_list
                else:
                    preds = np.append(preds, batch_pred_list, axis=0)
                batch_test_items = []
            answers.append(self.test_dict[test_items[i-1]])

        if len(batch_test_items) > 0:
            rating_pred = self.batch_predict(np.array(batch_test_items))
            rating_pred = rating_pred.cpu().data.numpy().copy()
            rating_pred[self.train_spmat[batch_test_items].toarray() > 0] = 0
            ind = np.argpartition(rating_pred, -40)[:, -40:]
            arr_ind = rating_pred[np.arange(len(rating_pred))[:, None], ind]
            arr_ind_argsort = np.argsort(arr_ind)[np.arange(len(rating_pred)), ::-1]
            batch_pred_list = ind[np.arange(len(rating_pred))[:, None], arr_ind_argsort]
            if preds is None:
                preds = batch_pred_list
            else:
                preds = np.append(preds, batch_pred_list, axis=0)
            batch_test_items = []
        assert preds.shape[0] == len(answers)

        test_scores, _, _ = get_full_sort_score(eval_info, answers, preds, verbose=verbose)
        #print('after test local model: ', self.local_model[0].device)
        return test_scores, len(test_items)


    def train_error_and_loss(self):
        self.model.eval()
        num_batches = int(len(self.train_i2ipairs[0]) / self.args.batch_size) + 1
        train_auc = 0
        loss = 0
        for eachstep in range(num_batches):
            s_items, d_items, neg_items = self.sampler.next_batch()
            s_items = torch.tensor(s_items, dtype=torch.long).to(self.device)
            d_items = torch.tensor(d_items, dtype=torch.long).to(self.device)
            neg_items = torch.tensor(neg_items, dtype=torch.long).to(self.device)
            item_embeddings = self.model(self.train_graph.x, self.train_graph.edge_index)

            s_items_embs = item_embeddings[s_items, :]
            d_items_embs = item_embeddings[d_items, :]
            neg_items_embs = item_embeddings[neg_items, :]
            pos_preds = torch.sum(s_items_embs*d_items_embs, dim=-1)
            neg_preds = torch.sum(s_items_embs*neg_items_embs, dim=-1)

            loss += torch.sum(-torch.log(torch.sigmoid(pos_preds - neg_preds)+1e-24)) / pos_preds.shape[0]
            train_auc += torch.sum(
                    (torch.sign(pos_preds - neg_preds) + 1) / 2) / pos_preds.shape[0]
        train_auc /= num_batches
        return train_auc, loss, len(self.train_i2ipairs[0])


    
    def test_personalized_model(self, eval_info, verbose=False):
        self.model.eval()
        self.update_parameters(self.personalized_model_bar)
        test_items = list(self.test_dict.keys())
        batch_test_items = []
        preds = None
        answers = []
        for i in range(1, len(test_items)+1):
            batch_test_items.append(test_items[i-1])
            if i % self.args.batch_size == 0:
                rating_pred = self.batch_predict(np.array(batch_test_items))
                rating_pred = rating_pred.cpu().data.numpy().copy()
                rating_pred[self.train_spmat[batch_test_items].toarray() > 0] = 0
                ind = np.argpartition(rating_pred, -40)[:, -40:]
                arr_ind = rating_pred[np.arange(len(rating_pred))[:, None], ind]
                arr_ind_argsort = np.argsort(arr_ind)[np.arange(len(rating_pred)), ::-1]
                batch_pred_list = ind[np.arange(len(rating_pred))[:, None], arr_ind_argsort]
                if preds is None:
                    preds = batch_pred_list
                else:
                    preds = np.append(preds, batch_pred_list, axis=0)
                batch_test_items = []
            answers.append(self.test_dict[test_items[i-1]])

        if len(batch_test_items) > 0:
            rating_pred = self.batch_predict(np.array(batch_test_items))
            rating_pred = rating_pred.cpu().data.numpy().copy()
            rating_pred[self.train_spmat[batch_test_items].toarray() > 0] = 0
            ind = np.argpartition(rating_pred, -40)[:, -40:]
            arr_ind = rating_pred[np.arange(len(rating_pred))[:, None], ind]
            arr_ind_argsort = np.argsort(arr_ind)[np.arange(len(rating_pred)), ::-1]
            batch_pred_list = ind[np.arange(len(rating_pred))[:, None], arr_ind_argsort]
            if preds is None:
                preds = batch_pred_list
            else:
                preds = np.append(preds, batch_pred_list, axis=0)
            batch_test_items = []
        assert preds.shape[0] == len(answers)
        test_scores, _, _ = get_full_sort_score(eval_info, answers, preds, verbose=verbose)
        self.update_parameters(self.local_model)

        return test_scores, len(test_items)


    def train_error_and_loss_personalized_model(self):
        self.model.eval()
        num_batches = int(len(self.train_i2ipairs[0]) / self.args.batch_size) + 1
        train_auc = 0
        loss = 0
        self.update_parameters(self.personalized_model_bar)
        for eachstep in range(num_batches):
            s_items, d_items, neg_items = self.sampler.next_batch()
            s_items = torch.tensor(s_items, dtype=torch.long).to(self.device)
            d_items = torch.tensor(d_items, dtype=torch.long).to(self.device)
            neg_items = torch.tensor(neg_items, dtype=torch.long).to(self.device)
            item_embeddings = self.model(self.train_graph.x, self.train_graph.edge_index)

            s_items_embs = item_embeddings[s_items, :]
            d_items_embs = item_embeddings[d_items, :]
            neg_items_embs = item_embeddings[neg_items, :]
            pos_preds = torch.sum(s_items_embs*d_items_embs, dim=-1)
            neg_preds = torch.sum(s_items_embs*neg_items_embs, dim=-1)

            loss += torch.sum(-torch.log(torch.sigmoid(pos_preds - neg_preds)+1e-24)) / pos_preds.shape[0]
            train_auc += torch.sum(
                    (torch.sign(pos_preds - neg_preds) + 1) / 2) / pos_preds.shape[0]
        train_auc /= num_batches
        self.update_parameters(self.local_model)
        return train_auc, loss , len(self.train_i2ipairs[0])


    
    def save_model(self):
        model_path = os.path.join("models", self.dataset)
        if not os.path.exists(model_path):
            os.makedirs(model_path)
        torch.save(self.model, os.path.join(model_path, "user_" + self.usermodel_id + ".pt"))

    def load_model(self):
        model_path = os.path.join("models", self.dataset)
        self.model = torch.load(os.path.join(model_path, "server" + ".pt"))
    
    @staticmethod
    def model_exists():
        return os.path.exists(os.path.join("models", "server" + ".pt"))
