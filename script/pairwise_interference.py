"""
Training (with SFT) on one query will affect greedy policy on other queries. 
We want to understand the structure of this pairwise interference effect
't1_sft' contains relevant logic for data preparation. 
we will do a thorough, analysis of such pairwise effect, we'd pick 1K 6-query subsets from GSM8K train set
train model on each query for 100 steps, then evaluate full-train set greedy accuracy, record the greedy response therein
so we'd obtain 1k .json files in the end here
"""