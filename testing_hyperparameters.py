from utils.agent import *
from utils.dataset import *
import numpy as np
import torch

from IPython.display import clear_output

import sys
import traceback
import sys
import os

batch_size = 32
PATH="./datasets/"



train_loader2012, val_loader2012 = read_voc_dataset(download=False, year='2012')
train_loader2007, val_loader2007 = read_voc_dataset(download=False, year='2007')

agents_per_class = {}
datasets_per_class = sort_class_extract([train_loader2007, train_loader2012])
datasets_eval_per_class = sort_class_extract([val_loader2007, val_loader2012])

classes = ['dog']

"""
Hyper-parameters à tester :
- ALPHA
- THRESHOLD
- NU 
def __init__(self, classe, alpha=0.2, nu=3.0, threshold=0.5, num_episodes=15, load=False )
"""
import pickle

alphas = [0.1, 0.5, 0.8]
nus = np.arange[2, 5, 10]
thresholds = [0.1, 0.4, 0.9]
results = {}
for alpha in alphas:
    results[str(alpha)] = {}
    for nu in nus:
        results[str(alpha)][str(nu)] = {}
        for threshold in thresholds:
            results[str(alpha)][str(nu)][str(threshold)] = []

count = 0
# 调试不同的超参数组合
for alpha in alphas:
    for nu in nus:
        for threshold in thresholds:
            for key in classes:
                count += 1
                print("Percentage =" + str((count) / (len(alphas) + len(nus) + len(thresholds))))
                agents_per_class[key] = Agent(key, alpha=alpha, nu=nu, num_episodes=15, threshold=threshold, load=False)
                agents_per_class[key].train(datasets_per_class[key])

                result = agents_per_class[key].evaluate(datasets_eval_per_class[key])   # 获取评估指标
                del agents_per_class[key]
                torch.cuda.empty_cache()
                results[str(alpha)][str(nu)][str(threshold)] = result   # 计算当前超参数下的评估指标(学习率 奖励值 epsilon用到的阈值)
                with open('hyperparameters_results.pickle', 'wb') as handle:
                    pickle.dump(results, handle, protocol=pickle.HIGHEST_PROTOCOL)  # 保存文件

print("Results : \n" + str(results))
