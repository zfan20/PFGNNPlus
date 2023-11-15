# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import json
from torch.utils.data import DataLoader
import numpy as np
import copy
import os
import time
from torch.optim import Adam
from tqdm import tqdm
import torch
from collections import OrderedDict
from utils.metrics import get_full_sort_score
from utils.sampler import WarpSampler


class IndUser:

    def __init__(self, args, train_spmat, train_dict, train_graph, test_dict, itemnum, device, model):
        self.args = args
        self.itemnum = itemnum
        self.train_graph = train_graph.to(device)
        self.train_spmat = train_spmat
        self.train_dict = train_dict
        self.train_i2ipairs = self.train_spmat.nonzero()
        self.test_dict = test_dict
        self.device = device
        self.model = model.to(device)
        self.sampler = WarpSampler(self.train_i2ipairs, self.train_dict, self.itemnum, batch_size=self.args.batch_size, n_workers=3)
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.args.learning_rate)
        self.losses = []

    def train(self):
        self.model.train()
        num_batches = int(len(self.train_i2ipairs[0]) / self.args.batch_size) + 1
        for epoch in range(1, self.args.local_epochs+1):
            avg_loss = 0.0
            start_time = time.time()
            #for s_items, d_items, neg_items in tqdm(self.trainloader):
            #    s_items = s_items.to(self.device)
            #    d_items = d_items.to(self.device)
            #    neg_items = neg_items.to(self.device)
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
            end_time = time.time()
            print("Epoch: ", epoch, ' has loss: ', avg_loss / num_batches, " with elapsed time: ", (end_time - start_time), ' seconds', flush=True)
            self.losses.append(avg_loss / num_batches)
        self.save_model()


    def predict(self):
        test_items = torch.tensor(np.array(test_dict.keys()), dtype=torch.long).to(self.device)
        self.model.eval()
        item_embeddings = self.model(self.train_graph.x, self.train_graph.edge_index)

        return torch.matmul(item_embeddings[test_items], item_embeddings.transpose(0, 1))


    def batch_predict(self, test_items):
        test_items = torch.tensor(np.array(test_items), dtype=torch.long).to(self.device)
        item_embeddings = self.model(self.train_graph.x, self.train_graph.edge_index)

        return torch.matmul(item_embeddings[test_items], item_embeddings.transpose(0, 1))

    def test(self, verbose=True):
        test_items = list(self.test_dict.keys())
        batch_test_items = []
        self.model.eval()
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
        return get_full_sort_score('Last Epoch', answers, preds, verbose=verbose)


    def save_model(self):
        self.model = self.model.to("cpu")
        model_path = os.path.join("models", self.args.dataset)                             
        if not os.path.exists(model_path):                                            
            os.makedirs(model_path)
        torch.save(self.model.state_dict(), os.path.join(model_path, self.args.algorithm+"_"+str(self.args.gnnlayers)+"_"+str(self.args.out_dim)+"_"+self.args.edge_type+"_"+str(self.args.weight_decay)+"_" + str(self.args.market)+"_induser.pt"))
        self.model = self.model.to(self.device)
                                                                                      
    def load_model(self): 
        model_path = os.path.join("models", self.args.dataset) 
        #loaded_state_dict = torch.load(os.path.join(model_path, self.args.algorithm+"_"+str(self.args.gnnlayers)+"_"+str(self.args.out_dim)+"_"+self.args.edge_type+"_"+str(self.args.weight_decay)+"_" + str(self.args.market)+"_induser.pt"), map_location='cpu')
        #state_dict = self.model.state_dict()
        #loaded_keys = list(loaded_state_dict.keys())
        #now_keys = list(state_dict.keys())
        #for keyA in state_dict:
        #    if 'weight' in keyA:
        #        for eachkey in loaded_keys:
        #            if 'weights' in eachkey:
        #                state_dict[keyA] = loaded_state_dict[eachkey]
        #                break
        #    else:
        #        for eachkey in loaded_keys:
        #            if 'bias' in eachkey:
        #                state_dict[keyA] = loaded_state_dict[eachkey]
        #                break
        #self.model.load_state_dict(state_dict)
        #self.model.to(self.device)
        self.model.load_state_dict(torch.load(os.path.join(model_path, self.args.algorithm+"_"+str(self.args.gnnlayers)+"_"+str(self.args.out_dim)+"_"+self.args.edge_type+"_"+str(self.args.weight_decay)+"_" + str(self.args.market)+"_induser.pt")))
        self.model.to(self.device)


    def load_from_pretrained_model(self, pretrain_dataset, pretrain_market, pretrain_algorithm):
        model_path = os.path.join("models", pretrain_dataset)
        #loaded_state_dict = torch.load(os.path.join(model_path, pretrain_algorithm+"_"+str(self.args.gnnlayers)+"_"+str(self.args.out_dim)+"_"+self.args.edge_type+"_"+str(self.args.weight_decay)+"_" + str(pretrain_market)+"_induser.pt"), map_location='cpu')
        #state_dict = self.model.state_dict()
        #loaded_keys = list(loaded_state_dict.keys())
        #now_keys = list(state_dict.keys())
        #for keyA in state_dict:
        #    if 'weight' in keyA:
        #        for eachkey in loaded_keys:
        #            if 'weight' in eachkey:
        #                print('change ', eachkey, ' to ', keyA)
        #                state_dict[keyA] = loaded_state_dict[eachkey]
        #                break
        #    else:
        #        for eachkey in loaded_keys:
        #            if 'bias' in eachkey:
        #                print('change ', eachkey, ' to ', keyA)
        #                state_dict[keyA] = loaded_state_dict[eachkey]
        #                break
        self.model.load_state_dict(torch.load(os.path.join(model_path, pretrain_algorithm+"_"+str(self.args.gnnlayers)+"_"+str(self.args.out_dim)+"_"+self.args.edge_type+"_"+str(self.args.weight_decay)+"_" + str(pretrain_market)+"_induser.pt")))
        self.model.to(self.device)


    def save_item_embs(self, path):
        item_embeddings = self.model(self.train_graph).cpu().data.numpy().copy()
        itemind_list = np.arange(0, self.itemnum)
        itemind_list = np.transpose(np.reshape(itemind_list, (1, -1)))

        output_item_embeddings = np.concatenate((itemind_list, item_embeddings), axis=1)
        np.savez(path+'_trained_item_emb', output_item_embeddings)
        print('save all embeddings as [itemind, emb_dims] to ', path+'_trained_item_emb')

                                                                                      
    @staticmethod                                                                     
    def model_exists():                            
        model_path = os.path.join("models", self.args.dataset)
        return os.path.exists(os.path.join(model_path, self.args.algorithm+"_"+str(self.args.gnnlayers)+"_"+str(self.args.out_dim)+"_"+self.args.edge_type+"_"+str(self.args.weight_decay)+"_" + str(self.args.market)+"_induser.pt"))
