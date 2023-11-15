# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0


import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import json
from tqdm import tqdm
import numpy as np
from torch.utils.data import DataLoader
from FLAlgorithms.optimizers.fedoptimizer import pFedMeOptimizer
from FLAlgorithms.users.userbase import User
from utils.metrics import get_full_sort_score
import copy

# Implementation for pFeMe clients

class UserpFedMeStruct(User):
    def __init__(self, device, usermodel_id, train_dict, train_spmat, train_graph, test_dict, itemnum, model, args):
        super().__init__(device, usermodel_id, train_dict, train_spmat, train_graph, test_dict, itemnum, model[0], args)
        """
        This user client only takes the vgae model with pooling on latent variables
        """

        self.K = args.K
        self.personal_learning_rate = args.personal_learning_rate
        self.personal_learning_rate_struct = args.personal_learning_rate_struct
        self.lamda_struct = args.lamda_struct

        self.optimizer = pFedMeOptimizer(self.model.parameters(), lr=self.personal_learning_rate, lamda=self.lamda)
        self.optimizerStruct = pFedMeOptimizer(self.model.parameters(), lr=self.personal_learning_rate_struct, lamda=self.lamda_struct)
        self.model_dir = os.path.join("models", self.args.dataset)
        self.model_path = os.path.join(self.model_dir, self.args.algorithm+"_"+str(self.args.gnnlayers)+"_"+str(self.args.out_dim)+"_"+self.args.edge_type+"_"+str(self.args.weight_decay)+"_" + str(self.args.market)+"_"+str(self.K)+"_"+str(self.personal_learning_rate)+"_"+str(self.usermodel_id)+"_"+str(self.args.learning_rate)+"_"+str(self.args.times)+"_"+str(self.args.numusers)+"_"+str(self.args.beta)+"_"+str(self.args.lamda)+"_"+str(self.args.batch_size)+"_"+str(self.args.local_epochs)+str(self.args.num_clusters)+"_"+str(self.args.num_clusters_level1)+"_"+str(self.args.pooldropout)+"_"+str(self.args.pool_kl_loss_weight)+"_"+str(self.args.heads)+"_"+str(self.args.tau)+"_"+str(self.args.pool_lp_loss_weight)+"_"+str(self.args.personal_learning_rate_struct)+"_"+str(self.args.lamda_struct)+"_pfedmestructuser.pt")

    def set_grads(self, new_grads):
        if isinstance(new_grads, nn.Parameter):
            for model_grad, new_grad in zip(self.model.parameters(), new_grads):
                model_grad.data = new_grad.data
        elif isinstance(new_grads, list):
            for idx, model_grad in enumerate(self.model.parameters()):
                model_grad.data = new_grads[idx]

    def batch_predict(self, test_items):
        test_items = torch.tensor(np.array(test_items), dtype=torch.long).to(self.device)
        self.model = self.model.to(self.device)
        self.model.eval()
        item_embeddings, _, _ = self.model(self.train_graph.x, self.train_graph.edge_index)

        preds = torch.matmul(item_embeddings[test_items], item_embeddings.transpose(0, 1))
        return preds
    
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
            item_embeddings, _, _ = self.model(self.train_graph.x, self.train_graph.edge_index)

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
            item_embeddings, _, _ = self.model(self.train_graph.x, self.train_graph.edge_index)

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
    

    def train(self):
        losses = []
        pool_kl_losses = []
        pool_lp_losses = []
        num_batches = int(len(self.train_i2ipairs[0]) / self.args.batch_size) + 1
        self.model.train()
        for epoch in range(1, self.local_epochs + 1):
            avg_loss = 0.0
            avg_pool_kl_loss = 0.0
            avg_pool_lp_loss = 0.0

            self.model.mu_encoder.lin.requires_grad = True
            self.model.logstd_encoder.lin.requires_grad = True
            self.model.mem1_mu.k.requires_grad = False
            self.model.mem1_logstd.k.requires_grad = False
            for eachstep in range(num_batches):
                s_items, d_items, neg_items = self.sampler.next_batch()
                s_items = torch.tensor(s_items, dtype=torch.long).to(self.device)
                d_items = torch.tensor(d_items, dtype=torch.long).to(self.device)
                neg_items = torch.tensor(neg_items, dtype=torch.long).to(self.device)
                self.optimizer.zero_grad()
                mu_embs, logstd_embs = self.model.encode(self.train_graph.x, self.train_graph.edge_index)
                unpooled_item_embeddings = self.model.reparametrize(mu_embs, logstd_embs)
                #unpooled_item_embeddings, level2_clusters_embs_array, pool_kl_loss = self.model(self.train_graph.x, self.train_graph.edge_index)
                kl_loss = self.model.kl_loss()

                s_items_embs = unpooled_item_embeddings[s_items, :]
                d_items_embs = unpooled_item_embeddings[d_items, :]
                neg_items_embs = unpooled_item_embeddings[neg_items, :]
                pos_preds = torch.sum(s_items_embs*d_items_embs, dim=-1)
                neg_preds = torch.sum(s_items_embs*neg_items_embs, dim=-1)

                loss = torch.sum(-torch.log(torch.sigmoid(pos_preds - neg_preds)+1e-24)) / pos_preds.shape[0] + 1/self.itemnum * kl_loss
                loss.backward()
                update_return, _ = self.optimizer.step(self.local_model)
                self.personalized_model_bar = copy.deepcopy(update_return)
                avg_loss += loss.item() / (num_batches)
            for new_param, localweight in zip(self.personalized_model_bar, self.local_model):
                localweight.data = localweight.data - self.lamda* self.learning_rate * (localweight.data - new_param.data)

            #kl_loss_allpool = 0.0
            self.model.mem1_mu.k.requires_grad = True
            self.model.mem1_logstd.k.requires_grad = True
            self.model.mu_encoder.lin.requires_grad = False
            self.model.logstd_encoder.lin.requires_grad = False
            num_items_perbatch = 4096
            for start_i in range(0, self.itemnum, num_items_perbatch):
                self.optimizerStruct.zero_grad()
                end_i = start_i + num_items_perbatch - 1
                if end_i + 1 >= self.itemnum:
                    end_i = self.itemnum - 1
                link_pred_items = torch.arange(start_i, end_i+1, dtype=torch.long).to(self.device)
                actual_links = torch.Tensor(self.train_spmat[start_i:end_i+1, :].todense()).to(self.device)
                _, _, pool_kl_loss = self.model(self.train_graph.x, self.train_graph.edge_index)
                lp_loss = self.model.pool_lploss(link_pred_items, actual_links)
                kl_loss_allpool = self.args.pool_kl_loss_weight * pool_kl_loss
                total_loss = self.args.pool_lp_loss_weight * lp_loss + kl_loss_allpool
                total_loss.backward()
                #kl_loss_allpool.backward()
                update_return, _ = self.optimizerStruct.step(self.local_model)
                self.personalized_model_bar = copy.deepcopy(update_return)

                avg_pool_kl_loss = self.args.pool_kl_loss_weight * pool_kl_loss.item()
                avg_pool_lp_loss = self.args.pool_lp_loss_weight * lp_loss.item() / (int(self.itemnum / num_items_perbatch)+1)
            #avg_pool_kl_loss = 0.0

            for new_param, localweight in zip(self.personalized_model_bar, self.local_model):
                localweight.data = localweight.data - self.lamda_struct* self.learning_rate * (localweight.data - new_param.data)

            losses.append(avg_loss)
            pool_kl_losses.append(avg_pool_kl_loss)
            pool_lp_losses.append(avg_pool_lp_loss)

        #update local model as local_weight_upated
        #self.clone_model_paramenter(self.local_weight_updated, self.local_model)
        self.update_parameters(self.local_model)
        #self.model_clusters_embs.weight = self.local_clusters_embs.weight
        #print('after train update personalize: ', self.personalized_model_bar)
        #print('after train update local: ', self.local_model)
        #print('after train update self model: ', list(self.model.parameters()))
        return losses[-1], pool_kl_losses[-1], pool_lp_losses[-1]


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
