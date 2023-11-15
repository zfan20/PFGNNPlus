# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import torch
import os
import numpy as np
from tqdm import tqdm
import copy

class Server:
    def __init__(self, device, dataset, algorithm, model, args):

        # Set up the main attributes
        self.device = device
        self.dataset = dataset
        self.users_datanames = []
        if self.dataset == 'multidomains':
            meta_dir = '../raw_data/meta_data/'
            meta_data_filelist = [f for f in os.listdir(meta_dir)]
            for eachcategory_filepath in meta_data_filelist:
                dataname = eachcategory_filepath.split('.')[0][5:]
                if dataname == 'Sports_and_Outdoors' or dataname == 'Books':
                    continue
                self.users_datanames = [dataname]
        elif self.dataset == 'multimarkets':
            #markets = ['ae', 'au', 'br', 'ca', 'cn', 'de', 'es', 'fr', 'in', 'jp', 'mx', 'nl', 'sa', 'sg', 'tr', 'uk', 'us']
            markets = ['us', 'ca', 'mx', 'uk', 'au', 'br','cn', 'de', 'es', 'fr', 'in', 'jp', 'nl', 'sa', 'sg', 'tr', 'ae', 'it']

            #remove markets without any data
            if args.category == 'Electronics':
                markets.remove('it') #market it has no data
            if args.category == 'Home_and_Kitchen':
                markets.remove('nl') #market nl has no data
            #markets = ['ae', 'au', 'br']
            #markets = ['ae', 'br']
            for eachmarket in markets:
                self.users_datanames.append((args.category, eachmarket))
        self.args = args
        self.num_glob_iters = args.num_global_iters
        self.local_epochs = args.local_epochs
        self.batch_size = args.batch_size
        self.learning_rate = args.learning_rate
        self.total_train_samples = 0
        self.model = copy.deepcopy(model).to(self.device)
        self.users = []
        self.selected_users = []
        self.num_users = args.numusers
        self.beta = args.beta
        self.lamda = args.lamda
        self.algorithm = args.algorithm
        self.times = args.times
        self.pathprefix = str(self.dataset)
        for param in [str(self.num_glob_iters), str(self.local_epochs), str(self.batch_size), str(self.learning_rate), str(self.num_users), str(self.beta), str(self.lamda), str(self.algorithm), str(self.times), str(self.args.personal_learning_rate), str(self.times), str(self.args.gnnlayers), str(self.args.out_dim), str(self.args.edge_type), str(self.args.market), str(self.args.model), str(self.args.dataset), str(self.args.category)]:
            self.pathprefix += '_' + param
            
    def aggregate_grads(self):
        assert (self.users is not None and len(self.users) > 0)
        for param in self.model.parameters():
            param.grad = torch.zeros_like(param.data)
        for user in self.users:
            self.add_grad(user, len(user.train_i2ipairs[0]) / self.total_train_samples)

    def add_grad(self, user, ratio):
        user_grad = user.get_grads()
        for idx, param in enumerate(self.model.parameters()):
            param.grad = param.grad + user_grad[idx].clone() * ratio

    def send_parameters(self):
        assert (self.users is not None and len(self.users) > 0)
        for user in self.users:
            user.set_parameters(self.model)

    def add_parameters(self, user, ratio):
        model = self.model.parameters()
        for server_param, user_param in zip(self.model.parameters(), user.get_parameters()):
            server_param.data = server_param.data + user_param.data.clone().to(self.device) * ratio

    def aggregate_parameters(self):
        assert (self.users is not None and len(self.users) > 0)
        for param in self.model.parameters():
            param.data = torch.zeros_like(param.data)
        total_train = 0
        for user in self.selected_users:
            total_train += len(user.train_i2ipairs[0])
        for user in self.selected_users:
            self.add_parameters(user, len(user.train_i2ipairs[0]) / total_train)

    def save_model(self):
        self.model = self.model.to("cpu")
        model_path = os.path.join("models", self.dataset)
        if not os.path.exists(model_path):
            os.makedirs(model_path)
        torch.save(self.model.state_dict(), os.path.join(model_path, self.pathprefix+"_server" + ".pt"))

    def load_model(self):
        model_path = os.path.join("models", self.dataset, self.pathprefix+"_server" + ".pt")
        assert (os.path.exists(model_path))
        self.model.load_state_dict(torch.load(model_path))
        self.model.to(self.device)

    def model_exists(self):
        return os.path.exists(os.path.join("models", self.dataset, self.pathprefix+"_server" + ".pt"))
    
    def select_users(self, round, num_users):
        '''selects num_clients clients weighted by number of samples from possible_clients
        Args:
            num_clients: number of clients to select; default 20
                note that within function, num_clients is set to
                min(num_clients, len(possible_clients))
        
        Return:
            list of selected clients objects
        '''
        if(num_users == len(self.users)):
            print("All users are selected")
            return self.users

        num_users = min(num_users, len(self.users))
        #np.random.seed(round)
        return np.random.choice(self.users, num_users, replace=False) #, p=pk)

    # define function for personalized agegatation.
    def personalized_update_parameters(self,user, ratio):
        # only argegate the local_weight_update
        for server_param, user_param in zip(self.model.parameters(), user.local_weight_updated):
            server_param.data = server_param.data + user_param.data.clone() * ratio


    def personalized_aggregate_parameters(self):
        assert (self.users is not None and len(self.users) > 0)

        # store previous parameters
        previous_param = copy.deepcopy(list(self.model.parameters()))
        for param in self.model.parameters():
            param.data = torch.zeros_like(param.data)
        total_train = 0
        #if(self.num_users = self.to)
        for user in self.selected_users:
            total_train += len(user.train_i2ipairs[0])

        for user in self.selected_users:
            self.add_parameters(user, len(user.train_i2ipairs[0]) / total_train)
            #self.add_parameters(user, 1 / len(self.selected_users))

        # aaggregate avergage model with previous model using parameter beta 
        for pre_param, param in zip(previous_param, self.model.parameters()):
            param.data = (1 - self.beta)*pre_param.data + self.beta*param.data
            
    # Save loss, accurancy to h5 fiel
    def save_results(self):
        pass

    def test(self, verbose):
        '''tests self.latest_model on given clients
        '''
        global_test_results = {}
        for c in self.users:
            c_score, num_test_items = c.test('Global Model Evaluated on '+c.usermodel_id, verbose)
            global_test_results[c.usermodel_id] = [c_score, num_test_items]

        return global_test_results

    def train_error_and_loss(self):
        global_train_error = {}
        for c in self.users:
            one_auc, one_loss, one_num_i2ipairs = c.train_error_and_loss() 
            global_train_error[c.usermodel_id] = [one_auc, one_loss, one_num_i2ipairs]

        return global_train_error

    def test_personalized_model(self, verbose):
        '''tests self.latest_model on given clients
        '''
        personalized_test_results = {}
        for c in self.users:
            c_score, num_test_items = c.test_personalized_model('Personalized Model Evaluated on '+c.usermodel_id, verbose)
            personalized_test_results[c.usermodel_id] = [c_score, num_test_items]

        return personalized_test_results

    def train_error_and_loss_personalized_model(self):
        personalized_train_error = {}
        for c in self.users:
            one_auc, one_loss, one_num_i2ipairs = c.train_error_and_loss_personalized_model() 
            personalized_train_error[c.usermodel_id] = [one_auc, one_loss, one_num_i2ipairs]
        
        return personalized_train_error

    def evaluate(self):
        stats_dict = self.test(verbose=False)  
        stats_train_dict = self.train_error_and_loss()
        all_usermodel_ids = list(stats_dict.keys())

        print('-'*20)
        print('Global Evaluation on all parties')
        for usermodel_id in all_usermodel_ids:
            auc = stats_train_dict[usermodel_id][0].item()
            loss = stats_train_dict[usermodel_id][1].item()
            num_pairs = stats_train_dict[usermodel_id][2]
            test_score = stats_dict[usermodel_id][0]
            num_test_items = stats_dict[usermodel_id][1]
            print('For Global ', str(usermodel_id), ' train auc: ', str(auc), ' train loss: ', str(loss), ' and has ', str(num_pairs), ' pairs', ' num test items: ', str(num_test_items), ' test scores: ', str(test_score), flush=True)
        print('-'*20)


    def evaluate_personalized_model(self):
        stats_dict = self.test_personalized_model(verbose=False)  
        stats_train_dict = self.train_error_and_loss_personalized_model()
        all_usermodel_ids = list(stats_dict.keys())
        print('-'*20)
        print('Personalize Evaluation on all parties')
        for usermodel_id in all_usermodel_ids:
            auc = stats_train_dict[usermodel_id][0].item()
            loss = stats_train_dict[usermodel_id][1].item()
            num_pairs = stats_train_dict[usermodel_id][2]
            test_score = stats_dict[usermodel_id][0]
            num_test_items = stats_dict[usermodel_id][1]
            print('For Personalized ', str(usermodel_id), ' train auc: ', str(auc), ' train loss: ', str(loss), ' and has ', str(num_pairs), ' pairs', ' num test items: ', str(num_test_items), ' test scores: ', str(test_score))
        print('-'*20, flush=True)

    def evaluate_one_step(self):
        for c in self.users:
            c.train_one_step()

        stats_dict = self.test()  
        stats_train_dict = self.train_error_and_loss()

        # set local model back to client for training process.
        for c in self.users:
            c.update_parameters(c.local_model)

        print('-'*20)
        print('Global-one-step Evaluation on all parties')
        for usermodel_id in all_usermodel_ids:
            auc = stats_train_dict[usermodel_id][0].item()
            loss = stats_train_dict[usermodel_id][1].item()
            num_pairs = stats_train_dict[usermodel_id][2]
            test_score = stats_dict[usermodel_id][0]
            num_test_items = stats_dict[usermodel_id][1]
            print('For Global-one-step ', str(usermodel_id), ' train auc: ', str(auc), ' train loss: ', str(loss), ' and has ', str(num_pairs), ' pairs', ' num test items: ', str(num_test_items), ' test scores: ', str(test_score))
        print('-'*20)

    def end_samplers(self):
        for c in self.users:
            c.sampler.close()

