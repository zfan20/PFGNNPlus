# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0


import torch
import os

from FLAlgorithms.users.userpFedMeStruct import UserpFedMeStruct
from FLAlgorithms.servers.serverbase import Server
from utils.model_utils import read_item2item_data, get_mostfree_gpu, read_item2item_xmarketdata
import numpy as np
import copy
 
# Implementation for pFedMe Server

class pFedMeStruct(Server):
    def __init__(self, device,  dataset, algorithm, model, args):
        super().__init__(device, dataset,algorithm, model[0], args)

        self.args = args
        self.model_clusters_embs = torch.nn.Embedding(args.num_clusters, args.hidden_size).to(self.device)
        #for test
        #markets = ['us', 'ca', 'ae', 'br']
        #self.users_datanames = []
        #for eachmarket in markets:
        #    self.users_datanames.append((args.category, eachmarket))
        #Initialize data for all  users
        total_users = len(self.users_datanames)
        for i in range(total_users):
            if dataset == 'multidomains':
                category_name = self.users_datanames[i]
                usermodel_id = category_name + "_" + args.model
                train_graph, train_spmat, train_dict, test_dict, itemnum = read_item2item_data(category_name, self.args.edge_type)
            else:
                category_name, market_name = self.users_datanames[i]
                usermodel_id = category_name + "_" + market_name + "_" + args.model
                train_graph, train_spmat, train_dict, test_dict, itemnum = read_item2item_xmarketdata(category_name, market_name, self.args.edge_type)
            if len(train_graph.edge_index[0]) < 100:
                print(usermodel_id + ' with training edges less than 100, skip')
                total_users -= 1
                continue
            device = torch.device("cuda:"+str(get_mostfree_gpu())) if torch.cuda.is_available() else torch.device("cpu")
            user = UserpFedMeStruct(device, usermodel_id, train_dict, train_spmat, train_graph, test_dict, itemnum, model, args)
            self.users.append(user)
            self.total_train_samples += len(user.train_i2ipairs[0])

        for param in [str(self.args.num_clusters), str(self.args.num_clusters_level1), str(self.args.pooldropout), str(self.args.pool_kl_loss_weight), str(self.args.heads), str(self.args.tau), str(self.args.pool_lp_loss_weight), str(self.args.personal_learning_rate_struct), str(self.args.lamda_struct)]:
            self.pathprefix += '_' + param

        print("Number of users / total users:", self.num_users, " / " ,total_users)
        print("Finished creating pFedMeStruct server.")

    def send_grads(self):
        assert (self.users is not None and len(self.users) > 0)
        grads = []
        for param in self.model.parameters():
            if param.grad is None:
                grads.append(torch.zeros_like(param.data))
            else:
                grads.append(param.grad)
        for user in self.users:
            user.set_grads(grads)
    

    def train(self):
        loss = []
        for glob_iter in range(self.num_glob_iters):
            print("-------------Round number: ",glob_iter, " -------------")
            loss_ = 0
            pool_kl_loss_ = 0
            pool_lp_loss_ = 0
            self.send_parameters()

            # Evaluate gloal model on user for each interation
            print("Evaluate global model")
            print("")
            self.evaluate()
            self.evaluate_personalized_model()

            for ind, user in enumerate(self.users):
                print('Training: ', user.usermodel_id, flush=True)
                last_local_epoch_loss, pool_kl_loss, pool_lp_loss = user.train() #* user.train_samples
                loss_ += last_local_epoch_loss + pool_kl_loss + pool_lp_loss
                pool_kl_loss_ += pool_kl_loss
                pool_lp_loss_ += pool_lp_loss
                print('Finished Training ', user.usermodel_id, ' total loss is: ', last_local_epoch_loss, ' pool_klloss: ', pool_kl_loss, ' pool_lp_loss: ', pool_lp_loss)
            loss_ /= len(self.users)
            pool_kl_loss_ /= len(self.users)
            pool_lp_loss_ /= len(self.users)
            loss.append(loss_)
            print('Global Optimization total loss: ', loss_, ' pool_klloss: ', pool_kl_loss_, ' pool_lp_loss: ', pool_lp_loss_, flush=True)

            self.selected_users = self.select_users(self.args.numusers,self.num_users)
            #self.aggregate_parameters()
            self.personalized_aggregate_parameters()

        #print(loss)
        #self.save_results()
        self.save_model()
        self.save_personalized_models()

    def kstepsTrain(self, K):
        print("-------------Load All Models-------------")
        self.load_all_models()
        print("-------------K Steps FineTune-------------")
        loss = []
        #self.send_parameters()
        for glob_iter in range(K):
            print("-------------Round number: ",glob_iter, " -------------")
            loss_ = 0
            for user in self.users:
                last_local_epoch_loss = user.trainfewsteps(1) #* user.train_samples
                loss_ += last_local_epoch_loss
            #self.evaluate_personalized_model()
            loss_ /= len(self.users)
            loss.append(loss_)
            print('Global Optimization loss: ', loss_)
        self.evaluate_personalized_model()

    def load_all_models(self):
        self.load_model()
        self.load_personalized_models()

    def save_personalized_models(self):
        for eachuser in self.users:
            eachuser.save_model()

    def load_personalized_models(self):
        for eachuser in self.users:
            eachuser.load_model()
