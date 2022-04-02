import pickle

from utils.agent import *
from utils.dataset import *

from IPython.display import clear_output

import sys
import traceback
import sys
import os
import tqdm.notebook as tq
import seaborn as sns

batch_size = 8
PATH="./datasets/"
classes = [ 'cat', 'dog', 'bird', 'motorbike','diningtable', 'train', 'tvmonitor', 'bus', 'horse', 'car', 'pottedplant', 'person', 'chair', 'boat', 'bottle', 'bicycle', 'aeroplane', 'cow', 'sheep', 'sofa']   # 20类
agents_per_class = {}

train_data_loc = "./datasets/datasets_per_class.pkl"    # 训练集文件
test_data_loc = "./datasets/datasets_eval_per_class.pkl"    # 测试集文件
if os.path.exists(train_data_loc) and os.path.exists(test_data_loc):   # 导入训练集和测试集字典
    train_file = open(train_data_loc, "rb")
    datasets_per_class = pickle.load(train_file)
    test_file = open(test_data_loc, "rb")
    datasets_eval_per_class = pickle.load(test_file)
else:
    train_loader2007, val_loader2007 = read_voc_dataset(download=False, year='2007')    # 2007年训练数据和测试数据
    train_loader2012, val_loader2012 = read_voc_dataset(download=False, year='2012')  # 2012年训练数据和测试数据
    datasets_per_class = sort_class_extract([train_loader2007, train_loader2012])  # 获取训练集数据集字典
    datasets_eval_per_class = sort_class_extract([val_loader2007, val_loader2012])  # 获取测试集数据集字典
    # 保存数据集字典
    # 保存训练集字典
    train_file = open("./datasets/datasets_per_class.pkl", "wb")
    pickle.dump(datasets_per_class, train_file)
    train_file.close()
    # 保存测试集字典
    test_file = open("./datasets/datasets_eval_per_class.pkl", "wb")
    pickle.dump(datasets_eval_per_class, test_file)
    test_file.close()

print("Data dictionary is ok.")

for i in tq.tqdm(range(len(classes))):  # 进度条(大小是类别的大小)
    classe = classes[i]
    print("Classe " + str(classe) + "...")      # 当前的类别
    agents_per_class[classe] = Agent(classe, alpha=0.2, num_episodes=15, load=False)    # 训练不使用任何网络，load设置为False
    agents_per_class[classe].train(datasets_per_class[classe])
    del agents_per_class[classe]    # 清除没用的空间
    torch.cuda.empty_cache()    # 清除没用的临时变量


# F:\postgraduate\A--论文准备\论文\论文库\英文论文\DRL+目标检测\强化学习在目标检测中的有效性\Active-Object-Localization-Deep-Reinforcement-Learning-master\Active-Object-Localization-Deep-Reinforcement-Learning-master\utils\agent.py:243: UserWarning: volatile was removed and now has no effect. Use `with torch.no_grad():` instead.
#   volatile=True).type(Tensor)
