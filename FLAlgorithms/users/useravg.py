# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0


import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import json
import numpy as np
from torch.utils.data import DataLoader
from utils.metrics import get_full_sort_score
from utils.sampler import WarpSampler
from FLAlgorithms.users.userbase import User

# Implementation for FedAvg clients

class UserAVG(User):
    def __init__(self, device, usermodel_id, train_dict, train_spmat, train_graph, test_dict, itemnum, model, args):
        super().__init__(device, usermodel_id, train_dict, train_spmat, train_graph, test_dict, itemnum, model[0], args)

        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.learning_rate)
        self.model_dir = os.path.join("models", self.args.dataset)
        self.model_path = os.path.join(self.model_dir, self.args.algorithm+"_"+str(self.args.gnnlayers)+"_"+str(self.args.out_dim)+"_"+self.args.edge_type+"_"+str(self.args.weight_decay)+"_" + str(self.args.market)+"_"+str(self.usermodel_id)+"_"+str(self.args.learning_rate)+"_"+str(self.args.times)+"_"+str(self.args.numusers)+"_"+str(self.args.batch_size)+"_"+str(self.args.local_epochs)+"_fedavguser.pt")

    def set_grads(self, new_grads):
        if isinstance(new_grads, nn.Parameter):
            for model_grad, new_grad in zip(self.model.parameters(), new_grads):
                model_grad.data = new_grad.data
        elif isinstance(new_grads, list):
            for idx, model_grad in enumerate(self.model.parameters()):
                model_grad.data = new_grads[idx]

    def train(self):
        self.model.train()
        losses = []
        num_batches = int(len(self.train_i2ipairs[0]) / self.args.batch_size) + 1
        for epoch in range(1, self.local_epochs + 1):
            avg_loss = 0.0
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

                loss = torch.sum(-torch.log(torch.sigmoid(pos_preds - neg_preds)+1e-24)) / pos_preds.shape[0]
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                avg_loss += loss.item()
            self.clone_model_paramenter(self.model.parameters(), self.local_model)
            losses.append(avg_loss / num_batches)
        return losses[-1]


    def trainfewsteps(self, steps):
        self.model.train()
        losses = []
        num_batches = int(len(self.train_i2ipairs[0]) / self.args.batch_size) + 1
        for epoch in range(steps):
            avg_loss = 0.0
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

                loss = torch.sum(-torch.log(torch.sigmoid(pos_preds - neg_preds)+1e-24)) / pos_preds.shape[0]
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                avg_loss += loss.item()
            self.clone_model_paramenter(self.model.parameters(), self.local_model)
            self.clone_model_paramenter(self.model.parameters(), self.personalized_model_bar)
            losses.append(avg_loss / num_batches)
        return losses[-1]


    def test_personalized_model(self, eval_info, verbose=False):
        self.model.eval()
        self.update_parameters(self.local_model)
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

        return test_scores, len(test_items)

    def train_error_and_loss_personalized_model(self):
        self.model.eval()
        num_batches = int(len(self.train_i2ipairs[0]) / self.args.batch_size) + 1
        self.update_parameters(self.local_model)
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
        return train_auc, loss , len(self.train_i2ipairs[0])


    def save_model(self):
        self.model = self.model.to("cpu")
        #model_path = os.path.join("models", self.args.dataset)
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)
        #torch.save(self.model.state_dict(), os.path.join(model_path, self.args.algorithm+"_"+str(self.args.gnnlayers)+"_"+str(self.args.out_dim)+"_"+self.args.edge_type+"_"+str(self.args.weight_decay)+"_" + str(self.args.market)+"_"+str(self.K)+"_"+str(self.personal_learning_rate)+"_"+str(self.usermodel_id)+"_pfedmeuser.pt"))
        torch.save(self.model.state_dict(), self.model_path)

    def load_model(self):
        #model_path = os.path.join("models", self.args.dataset)
        #self.model.load_state_dict(torch.load(os.path.join(model_path, self.args.algorithm+"_"+str(self.args.gnnlayers)+"_"+str(self.args.out_dim)+"_"+self.args.edge_type+"_"+str(self.args.weight_decay)+"_" + str(self.args.market)+"_"+str(self.K)+"_"+str(self.personal_learning_rate)+"_"+str(self.usermodel_id)+"_pfedmeuser.pt"), map_location=self.device))
        self.model.load_state_dict(torch.load(self.model_path))
        self.model.to(self.device)

    @staticmethod
    def model_exists():
        #model_path = os.path.join("models", self.args.dataset)
        #return os.path.exists(os.path.join(model_path, self.args.algorithm+"_"+str(self.args.gnnlayers)+"_"+str(self.args.out_dim)+"_"+self.args.edge_type+"_"+str(self.args.weight_decay)+"_" + str(self.args.market)+"_"+str(self.K)+"_"+str(self.personal_learning_rate)+"_"+str(self.usermodel_id)+"_pfedmeuser.pt"))
        return os.path.exists(self.model_path)
