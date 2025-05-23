# [Under review] All Items are not Equal: Addressing the Imbalance of Item Knowledge in LLM for Recommendation

We propose DECORec, a novel approach designed to address the imbalance in item information and enhance recommendation performance. The proposed DECORec applies a contrastive decoding strategy to effectively supplement information and guide the model away from inaccurate predictions. The proposed method is implemented during the inference stage.

**This repository is built on [CoLLM](https://github.com/zyang1580/CoLLM). Please refer to CoLLM's "readme.md" for an overview of the code structure.**

## 1. Prepare the Dataset and requirements

### Datasets

MovieLens-1M :  https://grouplens.org/datasets/movielens/

Amazon-Books : https://cseweb.ucsd.edu/~jmcauley/datasets.html#amazon_reviews

### Data pre-processing
The data pre-processing is based on CoLLM. To construct context+ as proposed in DECORec, genre is used for ML-1M, and category is used for Amazon-Book. For constructing context-, please refer to the appendix of our paper for adversarial attributes and unrelated sentences generated by the LLM.

## 2. DECORec example command

### Step 1. Setting Up the Environment and Preparing Vicuna, Pretrained CF Model, and the Trained CoLLM by Following CoLLM.
---
### Step 2. Performing DECORec in the Inference Stage
```
CUDA_VISIBLE_DEVICES=0,1 WORLD_SIZE=2 nohup torchrun --nproc-per-node 2 --master_port=11139 train_collm_mf_din.py  --cfg-path=train_configs/collm_pretrain_mf_ood.yaml > /log_result.log &
```

### Acknowledgements
Our repository is built upon [CoLLM](https://arxiv.org/abs/2310.19488]) and [BinLLM](https://aclanthology.org/2024.acl-long.497/), and we sincerely appreciate the contributions of their authors.
