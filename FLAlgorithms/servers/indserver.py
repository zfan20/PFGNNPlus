# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import torch
import os
import numpy as np
import copy
from torch.utils.data import DataLoader, RandomSampler

from utils.model_utils import read_item2item_data, get_mostfree_gpu, read_item2item_xmarketdata, read_item2item_xmarketdata_seq
from utils.model_utils import read_item2item_data_fromreviewedges, read_item2item_xmarketdata_fromreviewedges
from FLAlgorithms.users.induser import IndUser
from FLAlgorithms.users.indsequser import IndSeqUser
from FLAlgorithms.users.indbigramuser import IndBiGramUser
from FLAlgorithms.users.indpopuser import IndPopularItemsUser
from FLAlgorithms.trainmodel.models import SASRecModel

class IndServer:
    def __init__(self, args, model):
        self.args = args

        self.device = torch.device("cuda:"+str(get_mostfree_gpu())) if torch.cuda.is_available() else torch.device("cpu")
        self.model = model

        if model[0] is not None:
            #models not transformer
            if self.args.market == 'none':
                train_graph, train_spmat, train_dict, test_dict, itemnum = read_item2item_data(self.args.dataset, self.args.edge_type)
                self.dataset_dir = '../preprocess_data/' + args.edge_type + '_edges/' + args.dataset + '_' + args.edge_type + '/'
            else:
                train_graph, train_spmat, train_dict, test_dict, itemnum = read_item2item_xmarketdata(self.args.dataset, self.args.market, self.args.edge_type)
                self.dataset_dir = '../xmarket_preprocess/' + args.edge_type + '_edges/' + args.dataset + '_' + args.edge_type + '/' + args.market + '/'
            self.indmodel = model[0]
            if len(train_graph.edge_index[0]) < 100:
                print('market with training edges less than 100')
                exit()
            self.ind_user = IndUser(self.args, train_spmat, train_dict, train_graph, test_dict, itemnum, self.device, self.indmodel)
            print('random initialization evaluation:')
            self.test()
            if args.from_pretrain:
                if self.args.market == 'none':
                    self.ind_user.load_from_pretrained_model(args.pretrain_dataset, 'none', args.pretrain_algorithm)
                    print('Pretraining is loaded from: ', args.pretrain_dataset, ' and market none and algorithm ', args.pretrain_algorithm)
                else:
                    self.ind_user.load_from_pretrained_model(args.pretrain_dataset, args.pretrain_market, args.pretrain_algorithm)
                    print('Pretraining is loaded from: ', args.pretrain_dataset, ' and market ', args.pretrain_market, ' and algorithm ', args.pretrain_algorithm)
        elif model[1] == 'transformer':
            #transformer
            item_features, train_spmat, train_seqs, test_dict, itemnum = read_item2item_xmarketdata_seq(self.args.dataset, self.args.market, self.args.edge_type)
            self.indmodel = SASRecModel(item_features, itemnum+1, args, self.device, maxlen=100, hidden_dropout_prob=0.3)
            self.ind_user = IndSeqUser(self.args, train_spmat, train_seqs, test_dict, itemnum, self.device, self.indmodel)
        elif model[1] == 'relbigram':
            if self.args.market == 'none':
                _, _, train_dict, test_dict, itemnum = read_item2item_data(self.args.dataset, self.args.edge_type)
            else:
                _, _, train_dict, test_dict, itemnum = read_item2item_xmarketdata(self.args.dataset, self.args.market, self.args.edge_type)
            self.ind_user = IndBiGramUser(self.args, train_dict, test_dict, itemnum)
        elif model[1] == 'popularity':
            if self.args.market == 'none':
                _, _, train_dict, test_dict, itemnum = read_item2item_data(self.args.dataset, self.args.edge_type)
            else:
                _, _, train_dict, test_dict, itemnum = read_item2item_xmarketdata(self.args.dataset, self.args.market, self.args.edge_type)
            num_edges = sum([len(edges) for _, edges in train_dict.items()])
            if num_edges < 100:
                print('market with training edges less than 100')
                exit()
            self.ind_user = IndPopularItemsUser(self.args, train_dict, test_dict, itemnum)
        elif model[1] == 'intbigram':
            if self.args.market == 'none':
                train_dict, test_dict, itemnum = read_item2item_data_fromreviewedges(self.args.dataset, self.args.edge_type)
            else:
                train_dict, test_dict, itemnum = read_item2item_xmarketdata_fromreviewedges(self.args.dataset, self.args.market, self.args.edge_type)
            self.ind_user = IndBiGramUser(self.args, train_dict, test_dict, itemnum)
        elif model[1] == 'intpopularity':
            if self.args.market == 'none':
                train_dict, test_dict, itemnum = read_item2item_data_fromreviewedges(self.args.dataset, self.args.edge_type)
            else:
                train_dict, test_dict, itemnum = read_item2item_xmarketdata_fromreviewedges(self.args.dataset, self.args.market, self.args.edge_type)
            num_edges = sum([len(edges) for _, edges in train_dict.items()])
            if num_edges < 100:
                print('market with training edges less than 100')
                exit()
            self.ind_user = IndPopularItemsUser(self.args, train_dict, test_dict, itemnum)
        #self.ind_user = IndUser(self.args, self.train_loader, train_spmat, train_dict, train_graph, test_dict, itemnum, self.device, self.indmodel)

    def train(self):
        self.ind_user.train()

    def test(self, verbose=True):
        scores, results_info, _ = self.ind_user.test(verbose=verbose)
        return scores
    
    def save_embs(self):
        self.ind_user.save_item_embs(self.dataset_dir+self.model[1])

    def end_samplers(self):
        self.ind_user.sampler.close()
