# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import torch
import os

from FLAlgorithms.users.userpFedMe import UserpFedMe
from FLAlgorithms.servers.serverbase import Server
from utils.model_utils import read_item2item_data, get_mostfree_gpu, read_item2item_xmarketdata
import numpy as np
 
# Implementation for pFedMe Server

class pFedMe(Server):
    def __init__(self, device,  dataset, algorithm, model, args):
        super().__init__(device, dataset,algorithm, model[0], args)

        self.args = args
        # Initialize data for all  users
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
            #id, train , test = read_user_data(i, data, dataset)
            #device = torch.device("cuda:"+str(get_mostfree_gpu())) if torch.cuda.is_available() else torch.device("cpu")
            #device = torch.device("cpu")
            device = torch.device("cuda:"+str(get_mostfree_gpu())) if torch.cuda.is_available() else torch.device("cpu")
            user = UserpFedMe(device, usermodel_id, train_dict, train_spmat, train_graph, test_dict, itemnum, model, args)
            self.users.append(user)
            self.total_train_samples += len(user.train_i2ipairs[0])

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
            self.send_parameters()

            # Evaluate gloal model on user for each interation
            print("Evaluate global model")
            print("")
            self.evaluate()
            self.evaluate_personalized_model()

            for ind, user in enumerate(self.users):
                last_local_epoch_loss = user.train() #* user.train_samples
                loss_ += last_local_epoch_loss
                print('Finished Training ', user.usermodel_id, ' total loss is: ', loss_)
            loss_ /= len(self.users)
            loss.append(loss_)
            print('Global Optimization total loss: ', loss_, flush=True)

            self.selected_users = self.select_users(self.args.numusers,self.num_users)
            #self.aggregate_parameters()
            self.personalized_aggregate_parameters()

        #print(loss)
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
