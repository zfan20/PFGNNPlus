# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

#!/usr/bin/env python
import matplotlib.pyplot as plt
import numpy as np
import argparse
import importlib
import random
import os
import time
from FLAlgorithms.servers.serveravg import FedAvg
from FLAlgorithms.servers.serverpFedMe import pFedMe
from FLAlgorithms.servers.serverpFedMeStruct import pFedMeStruct
from FLAlgorithms.servers.indserver import IndServer
from FLAlgorithms.trainmodel.models import *
from utils.model_utils import get_mostfree_gpu
from utils.plot_utils import *
import torch
from utils.metrics import set_seed

def main(args):

    # Get device status: Check GPU or CPU
    #device = torch.device("cuda:{}".format(gpu) if torch.cuda.is_available() and gpu != -1 else "cpu")
    device = torch.device("cuda:"+str(get_mostfree_gpu())) if torch.cuda.is_available() else torch.device("cpu")
    #device = torch.device("cpu")

    set_seed(args.random_seed)

    for i in range(args.times):
        print("---------------Running time:------------",i)
        start_time = time.time()
        # Generate model

        if args.model == 'sgcn':
            model = SGCN(args.gnnlayers, 768, args.out_dim), args.model

        if args.model == 'fedvarsgcn':
            model = FedVariationalSGCNEncoder(args), args.model

        if args.model == 'featmlp':
            model = FeatMLP(768, args.out_dim), args.model

        if args.model in ['transformer', 'relbigram', 'popularity', 'intbigram', 'intpopularity']:
            #model define should be after the data load, will be done in user or server
            model = None, args.model

        # select algorithm
        if args.algorithm == "FedAvg":
            server = FedAvg(device, args.dataset, args.algorithm, model, args)
        
        if args.algorithm == "pFedMe":
            server = pFedMe(device, args.dataset, args.algorithm, model, args)

        if args.algorithm == "pFedMeStruct":
            server = pFedMeStruct(device, args.dataset, args.algorithm, model, args)

        if args.algorithm != 'pretrain_test_no_finetune':
            if args.frompretrain_fewsteps:
                server.kstepsTrain(args.Ksteps)
            else:
                server.train()
            print('After all training', '-'*20)
        server.test(verbose=True)
        if not (args.algorithm == "IndTrain" or args.algorithm == 'pretrain_test_no_finetune' or args.algorithm == 'pretrain_test_with_finetune'):
            server.test_personalized_model(verbose=True)
        if args.save_item_embs:
            server.save_embs()

        end_time = time.time()
        print("Running time ", i, " used ", (end_time - start_time), " seconds")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="Cifar10", help='Dataset')
    parser.add_argument("--category", type=str, default="Electronics", help='category when multi-markets')
    parser.add_argument("--model", type=str, default="sgcn", choices=["dnn", "mclr", "cnn", "sgcn", "featmlp", "transformer", "relbigram", "popularity", "intbigram", "intpopularity", "fedvarsgcn"])
    parser.add_argument("--batch_size", type=int, default=1024)
    parser.add_argument("--learning_rate", type=float, default=0.001, help="Local learning rate")
    parser.add_argument("--beta", type=float, default=1.0, help="Average moving parameter for pFedMe, or Second learning rate of Per-FedAvg")
    parser.add_argument("--lamda", type=float, default=15, help="Regularization term")
    parser.add_argument("--num_global_iters", type=int, default=100)
    parser.add_argument("--local_epochs", type=int, default=100)
    parser.add_argument("--optimizer", type=str, default="SGD", choices=["SGD", "Adam"])
    parser.add_argument("--algorithm", type=str, default="pFedMe",choices=["pFedMe", "PerAvg", "FedAvg", "IndTrain", "pretrain_test_no_finetune", "pretrain_test_with_finetune", "pFedMeStruct"]) 
    parser.add_argument("--numusers", type=int, default=20, help="Number of Users per round")
    parser.add_argument("--K", type=int, default=5, help="Computation steps")
    parser.add_argument("--personal_learning_rate", type=float, default=0.09, help="Persionalized learning rate to caculate theta aproximately using K steps")
    parser.add_argument("--times", type=int, default=5, help="running time")
    parser.add_argument("--gpu", type=int, default=0, help="Which GPU to run the experiments, -1 mean CPU, 0,1,2 for GPU")

    parser.add_argument("--gnnlayers", type=int, default=3,
                        help="Number of GNN Layers")
    parser.add_argument("--out_dim", type=int, default=128,
                        help="The output dimension of items")
    parser.add_argument("--edge_type", type=str, default="also_buy",
                        help="The type of item-item edge")
    parser.add_argument("--weight_decay", type=float, default=1e-2,
                        help="weight regularization")
    parser.add_argument("--market", type=str, default='us',
                        help="market to run independently")
    parser.add_argument("--save_item_embs", action='store_true')

    #Transformer
    parser.add_argument("--hidden_size", type=int, default=128, help="hidden size of transformer model")
    parser.add_argument("--num_hidden_layers", type=int, default=1, help="number of layers")
    parser.add_argument('--num_attention_heads', default=2, type=int)
    parser.add_argument('--hidden_act', default="gelu", type=str) # gelu relu
    parser.add_argument("--attention_probs_dropout_prob", type=float, default=0.0, help="attention dropout p")
    parser.add_argument("--hidden_dropout_prob", type=float, default=0.3, help="hidden dropout p")
    parser.add_argument('--max_seq_length', default=100, type=int)

    
    #pretrained
    parser.add_argument('--from_pretrain', action='store_true')
    parser.add_argument('--pretrain_dataset', default='Electronics', type=str)
    parser.add_argument('--pretrain_market', default='ae', type=str)
    parser.add_argument('--pretrain_algorithm', default='IndTrain', type=str)


    #few-steps training
    parser.add_argument('--frompretrain_fewsteps', action='store_true')
    parser.add_argument('--Ksteps', default=1, type=int)

    #mempool params
    parser.add_argument('--num_clusters', default=6, type=int)
    parser.add_argument('--num_clusters_level1', default=24, type=int)
    parser.add_argument('--pool_kl_loss_weight', default=0.0, type=float)
    parser.add_argument('--pooldropout', default=0.1, type=float)
    parser.add_argument('--heads', default=2, type=int)
    parser.add_argument('--tau', default=1.0, type=float)
    parser.add_argument('--pool_lp_loss_weight', default=1.0, type=float)
    parser.add_argument('--personal_learning_rate_struct', default=0.01, type=float)
    parser.add_argument('--lamda_struct', default=1.0, type=float)

    parser.add_argument('--random_seed', default=0, type=int)

    args = parser.parse_args()

    print("=" * 80)
    print("Summary of training process:")
    print("Algorithm: {}".format(args.algorithm))
    print("Batch size: {}".format(args.batch_size))
    print("Learing rate       : {}".format(args.learning_rate))
    print("Average Moving       : {}".format(args.beta))
    print("Subset of users      : {}".format(args.numusers))
    print("Number of global rounds       : {}".format(args.num_global_iters))
    print("Number of local rounds       : {}".format(args.local_epochs))
    print("Dataset       : {}".format(args.dataset))
    print("Local Model       : {}".format(args.model))
    print("Market       : {}".format(args.market))
    if args.model == 'sgcn':
        print("#of GNN Layers       : {}".format(args.gnnlayers))
        print("Edge Type is       : {}".format(args.edge_type))
        print("The output dimension       : {}".format(args.out_dim))
        print("The weight decay       : {}".format(args.weight_decay))
    print("=" * 80)

    main(args)
