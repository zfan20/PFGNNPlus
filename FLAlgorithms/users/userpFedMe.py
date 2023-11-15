# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0


import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import json
from tqdm import tqdm
from torch.utils.data import DataLoader
from FLAlgorithms.optimizers.fedoptimizer import pFedMeOptimizer
from FLAlgorithms.users.userbase import User
import copy

# Implementation for pFeMe clients

class UserpFedMe(User):
    def __init__(self, device, usermodel_id, train_dict, train_spmat, train_graph, test_dict, itemnum, model, args):
        super().__init__(device, usermodel_id, train_dict, train_spmat, train_graph, test_dict, itemnum, model[0], args)


        self.K = args.K
        self.personal_learning_rate = args.personal_learning_rate
        self.optimizer = pFedMeOptimizer(self.model.parameters(), lr=self.personal_learning_rate, lamda=self.lamda)
        self.model_dir = os.path.join("models", self.args.dataset)
        self.model_path = os.path.join(self.model_dir, self.args.algorithm+"_"+str(self.args.gnnlayers)+"_"+str(self.args.out_dim)+"_"+self.args.edge_type+"_"+str(self.args.weight_decay)+"_" + str(self.args.market)+"_"+str(self.K)+"_"+str(self.personal_learning_rate)+"_"+str(self.usermodel_id)+"_"+str(self.args.learning_rate)+"_"+str(self.args.times)+"_"+str(self.args.numusers)+"_"+str(self.args.beta)+"_"+str(self.args.lamda)+"_"+str(self.args.batch_size)+"_"+str(self.args.local_epochs)+"_pfedmeuser.pt")

    def set_grads(self, new_grads):
        if isinstance(new_grads, nn.Parameter):
            for model_grad, new_grad in zip(self.model.parameters(), new_grads):
                model_grad.data = new_grad.data
        elif isinstance(new_grads, list):
            for idx, model_grad in enumerate(self.model.parameters()):
                model_grad.data = new_grads[idx]

    def train(self):
        losses = []
        num_batches = int(len(self.train_i2ipairs[0]) / self.args.batch_size) + 1
        self.model.train()
        for epoch in range(1, self.local_epochs + 1):
            avg_loss = 0.0
            for eachstep in range(num_batches):
                s_items, d_items, neg_items = self.sampler.next_batch()
                s_items = torch.tensor(s_items, dtype=torch.long).to(self.device)
                d_items = torch.tensor(d_items, dtype=torch.long).to(self.device)
                neg_items = torch.tensor(neg_items, dtype=torch.long).to(self.device)
                self.optimizer.zero_grad()
                item_embeddings = self.model(self.train_graph.x, self.train_graph.edge_index)

                s_items_embs = item_embeddings[s_items, :]
                d_items_embs = item_embeddings[d_items, :]
                neg_items_embs = item_embeddings[neg_items, :]
                pos_preds = torch.sum(s_items_embs*d_items_embs, dim=-1)
                neg_preds = torch.sum(s_items_embs*neg_items_embs, dim=-1)

                loss = torch.sum(-torch.log(torch.sigmoid(pos_preds - neg_preds)+1e-24)) / pos_preds.shape[0]
                loss.backward()
                update_return, _ = self.optimizer.step(self.local_model)
                self.personalized_model_bar = copy.deepcopy(update_return)
                avg_loss += loss.item() / (num_batches)
            # update local weight after finding aproximate theta
            for new_param, localweight in zip(self.personalized_model_bar, self.local_model):
                localweight.data = localweight.data - self.lamda* self.learning_rate * (localweight.data - new_param.data)
            losses.append(avg_loss)

        #update local model as local_weight_upated
        #self.clone_model_paramenter(self.local_weight_updated, self.local_model)
        self.update_parameters(self.local_model)
        #print('after train update personalize: ', self.personalized_model_bar)
        #print('after train update local: ', self.local_model)
        #print('after train update self model: ', list(self.model.parameters()))
        return losses[-1]

    def trainfewsteps(self, steps):
        losses = []
        num_batches = int(len(self.train_i2ipairs[0]) / self.args.batch_size) + 1
        self.model.train()
        self.local_model = copy.deepcopy(list(self.model.parameters()))
        for epoch in range(steps):
            avg_loss = 0.0
            for eachstep in range(num_batches):
                s_items, d_items, neg_items = self.sampler.next_batch()
                s_items = torch.tensor(s_items, dtype=torch.long).to(self.device)
                d_items = torch.tensor(d_items, dtype=torch.long).to(self.device)
                neg_items = torch.tensor(neg_items, dtype=torch.long).to(self.device)
                self.optimizer.zero_grad()
                #print('edge index device: ', self.train_graph.edge_index.device, ' model device: ', next(self.model.parameters()).device)
                item_embeddings = self.model(self.train_graph.x, self.train_graph.edge_index)

                s_items_embs = item_embeddings[s_items, :]
                d_items_embs = item_embeddings[d_items, :]
                neg_items_embs = item_embeddings[neg_items, :]
                pos_preds = torch.sum(s_items_embs*d_items_embs, dim=-1)
                neg_preds = torch.sum(s_items_embs*neg_items_embs, dim=-1)

                loss = torch.sum(-torch.log(torch.sigmoid(pos_preds - neg_preds)+1e-24)) / pos_preds.shape[0]
                loss.backward()
                #self.personalized_model_bar, _ = self.optimizer.step(self.local_model)
                update_return, _ = self.optimizer.step(self.local_model)
                avg_loss += loss.item() / (num_batches)
            self.local_model = copy.deepcopy(update_return)
            losses.append(avg_loss)
        self.personalized_model_bar = copy.deepcopy(self.local_model)

        return losses[-1]

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
