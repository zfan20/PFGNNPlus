# Personalized federated domain adaptation for item-to-item recommendation (PFGNNPlus, UAI 2023)
Just got the approval to release the code and will update the codebase soon.

This is the implementation for PFGNNPlus UAI'23. You may find the paper on [Arxiv](https://arxiv.org/pdf/2306.03191.pdf).

## Abstract
Item-to-Item (I2I) recommendation is an important function that suggests replacement or complement options for an item based on their functional similarities or synergies. To capture such item relationships effectively, the recommenders need to understand why subsets of items are co-viewed or co-purchased by the customers. Graph-based models, such as graph neural networks (GNNs), provide a natural framework to combine, ingest and extract valuable insights from such high-order item relationships. However, learning GNNs effectively for I2I requires ingesting a large amount of relational data, which might not always be available, especially in new, emerging market segments. To mitigate this data bottleneck, we postulate that recommendation patterns learned from existing market segments (with private data) could be adapted to build effective warm-start models for emerging ones. To achieve this, we introduce a personalized graph adaptation model based on GNNs to summarize, assemble, and adapt recommendation patterns across market segments with heterogeneous customer behaviors into effective local models.


## Dataset
Please download the open public dataset from https://xmrec.github.io/

## Execution Environment
We suggest to use multi-gpus machine to run the algorithm.

## Example code to run
```
python main.py --dataset multimarkets --edge_type alsoViewed_alsoBought --model fedvarsgcn --numusers 5 --category Electronics --local_epochs 20 --algorithm pFedMeStruct --times 1 --num_global_iters 200 --personal_learning_rate 0.001 --beta 0.5 --lamda 1 --learning_rate 0.1 --heads 1 --num_clusters_level1 24 --pool_kl_loss_weight 1.0 --tau 0.001 --pool_lp_loss_weight 1.0 --personal_learning_rate_struct 0.001 --lamda_struct 1.0 --random_seed 2022
```

### Please cite our papers if you use the code:
```bibtex

@InProceedings{pmlr-v216-fan23a,
  title = 	 {Personalized federated domain adaptation for item-to-item recommendation},
  author =       {Fan, Ziwei and Ding, Hao and Deoras, Anoop and Hoang, Trong Nghia},
  booktitle = 	 {Proceedings of the Thirty-Ninth Conference on Uncertainty in Artificial Intelligence},
  pages = 	 {560--570},
  year = 	 {2023},
  editor = 	 {Evans, Robin J. and Shpitser, Ilya},
  volume = 	 {216},
  series = 	 {Proceedings of Machine Learning Research},
  month = 	 {31 Jul--04 Aug},
  publisher =    {PMLR},
  pdf = 	 {https://proceedings.mlr.press/v216/fan23a/fan23a.pdf},
  url = 	 {https://proceedings.mlr.press/v216/fan23a.html},
  abstract = 	 {Item-to-Item (I2I) recommendation is an important function that suggests replacement or complement options for an item based on their functional similarities or synergies. To capture such item relationships effectively, the recommenders need to understand why subsets of items are co-viewed or co-purchased by the customers. Graph-based models, such as graph neural networks (GNNs), provide a natural framework to combine, ingest and extract valuable insights from such high-order item relationships. However, learning GNNs effectively for I2I requires ingesting a large amount of relational data, which might not always be available, especially in new, emerging market segments. To mitigate this data bottleneck, we postulate that recommendation patterns learned from existing market segments (with private data) could be adapted to build effective warm-start models for emerging ones. To achieve this, we introduce a personalized graph adaptation model based on GNNs to summarize, assemble and adapt recommendation patterns across market segments with heterogeneous customer behaviors into effective local models.}
}
```

