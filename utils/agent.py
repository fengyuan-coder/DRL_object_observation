#import torchvision.datasets.SBDataset as sbd
from utils.models import *
from utils.tools import *
import os
import imageio
import math
import random
import numpy as np

import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.datasets as datasets

from itertools import count
from PIL import Image
import torch.optim as optim
import cv2 as cv
from torch.autograd import Variable

from tqdm.notebook import tqdm
from config import *

import glob
from PIL import Image

class Agent():
    def __init__(self, classe, alpha=0.2, nu=3.0, threshold=0.5, num_episodes=15, load=False ):
        """
            类初始化学习参数集时，代理与数据集中的给定类相关联。
        """
        self.BATCH_SIZE = 100   # 数据批次
        self.GAMMA = 0.900  # 折扣率
        self.EPS = 1    # epsilon,训练过程中前5次每次递减0.18; 不断提高最优动作选择的概率
        self.TARGET_UPDATE = 1
        self.save_path = SAVE_MODEL_PATH    # config文件中定义; 模型保存路径
        screen_height, screen_width = 224, 224  # 图片尺寸(长宽)
        self.n_actions = 9      # 共9种动作
        self.classe = classe

        self.feature_extractor = FeatureExtractor()     # 提取特征
        if not load:
            self.policy_net = DQN(screen_height, screen_width, self.n_actions)  # 策略网络 Π, 每个动作的打分
        else:
            self.policy_net = self.load_network()   # 调用现有的网络 作 策略网络
            
        self.target_net = DQN(screen_height, screen_width, self.n_actions)  # 定义目标网络
        self.target_net.load_state_dict(self.policy_net.state_dict())   # 用策略网络参数初始化目标网络
        self.target_net.eval()  # to not do dropout
        self.feature_extractor.eval()   # to not do dropout
        if use_cuda:
            # 将提取特征网络 目标网络和策略网络放到cuda中
            self.feature_extractor = self.feature_extractor.cuda()
            self.target_net = self.target_net.cuda()
            self.policy_net = self.policy_net.cuda()
        
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=1e-6)  # 优化器
        self.memory = ReplayMemory(10000)   # 缓存列表大小为10000
        self.steps_done = 0     # 回合动作数
        self.episode_durations = []
        
        self.alpha = alpha # [0, 1]  缩放因子
        self.nu = nu # Reward of Trigger
        self.threshold = threshold  # iou的阈值
        self.actions_history = []   # 执行过的动作
        self.num_episodes = num_episodes    # 每个类别训练15次
        self.actions_history += [[100]*9]*20    # 预设20组动作，初始值都是100

    def save_network(self):
        """
            保存Q-Network
        """
        torch.save(self.policy_net, self.save_path+"_"+self.classe)
        print('Saved')

    def load_network(self):
        """
            调用现有的Q网络
        """
        if not use_cuda:
            return torch.load(self.save_path+"_"+self.classe, map_location=torch.device('cpu'))
        return torch.load(self.save_path+"_"+self.classe)



    def intersection_over_union(self, box1, box2):
        """
            计算两个标注框的交并比

        """
        x11, x21, y11, y21 = box1
        x12, x22, y12, y22 = box2
        
        yi1 = max(y11, y12)
        xi1 = max(x11, x12)
        yi2 = min(y21, y22)
        xi2 = min(x21, x22)
        inter_area = max(((xi2 - xi1) * (yi2 - yi1)), 0)    # 两个标注框的交集面积
        box1_area = (x21 - x11) * (y21 - y11)   # 我们创建的标注框面积
        box2_area = (x22 - x12) * (y22 - y12)   # 真实标注框面积
        union_area = box1_area + box2_area - inter_area     # 两个标注框的并集面积

        iou = inter_area / union_area   # 交并比
        return iou

    def compute_reward(self, actual_state, previous_state, ground_truth):
        """
            通过两组标注框和真实标注框计算各自的交并比 并 求差
        """
        res = self.intersection_over_union(actual_state, ground_truth) - self.intersection_over_union(previous_state, ground_truth)
        if res <= 0:
            return -1   # 新添加的这个动作不好
        return 1    # 新添加的这个动作很好
      
    def rewrap(self, coord):
        # coord<=0, 0; coord>0, min(coord, 224)
        return min(max(coord,0), 224)
      
    def compute_trigger_reward(self, actual_state, ground_truth):
        """
            根据标注框和交并比计算奖励
        """
        res = self.intersection_over_union(actual_state, ground_truth)  # 计算我们的标注框和真实标注框的交并比
        if res>=self.threshold: # 交并比不低于0.5
            return self.nu  # 3
        return -1*self.nu   # 交并比低于0.5 -> -3

    def get_best_next_action(self, actions, ground_truth):
        """
            根据历史动作集actions判断所有动作的好坏(能否缩小标注框和真实标注框的差距), 有好动作就随机选一个好动作, 没有好动作就随机选一个动作
        """
        max_reward = -99
        best_action = -99
        positive_actions = []   # 好动作 -> 缩小选择标注框和真实标注框的差异
        negative_actions = []   # 坏动作 -> 增大选择标注框和真实标注框的差异
        actual_equivalent_coord = self.calculate_position_box(actions)  # 经过历史动作集actions后的标注框坐标
        for i in range(0, 9):
            copy_actions = actions.copy()   # 不使用拷贝会修改历史动作集actions
            copy_actions.append(i)  # 在历史动作集中添加一个动作([0,8])
            new_equivalent_coord = self.calculate_position_box(copy_actions)    # 经过copy_actions后的标注框坐标
            if i!=0:
                reward = self.compute_reward(new_equivalent_coord, actual_equivalent_coord, ground_truth)   # 1->i动作好, -1->i动作不好
            else:   # 动作0是停止标志
                reward = self.compute_trigger_reward(new_equivalent_coord,  ground_truth)   # 交并比和阈值比较, 不低于阈值->(3)/低于阈值->(-3)

            # 根据reward判断动作的好坏并保存
            if reward >= 0:
                positive_actions.append(i)
            else:
                negative_actions.append(i)
        if len(positive_actions) == 0:  # 没有好动作 -> 随机选一个动作
            return random.choice(negative_actions)
        return random.choice(positive_actions)  # 随机选一个好动作


    def select_action(self, state, actions, ground_truth):
        """
            根据状态选择操作
        """
        sample = random.random()
        eps_threshold = self.EPS
        self.steps_done += 1
        if sample > eps_threshold:  # (1-epsilon)的概率 根据策略网络选择最优动作
            with torch.no_grad():
                if use_cuda:
                    inpu = Variable(state).cuda()   # state
                else:
                    inpu = Variable(state)
                qval = self.policy_net(inpu)    # Π(a|s), 获得每个动作的打分
                # print("qval shape" + str(qval.shape))
                _, predicted = torch.max(qval.data, 1)  # 最优动作的索引; 1表示取行最值，返回最值和索引
                # print("predicted shape" + str(predicted.shape))
                action = predicted[0]   # 取出最佳的动作
                try:
                  return action.cpu().numpy()[0]    # 以numpy数组传到cpu, 多个动作选择第一个
                except:
                  return action.cpu().numpy()   # 以numpy数组传到cpu
        else:   # epsilon的概率 根据历史动作集 选择最优动作
            #return np.random.randint(0,9)   # 随机选择动作
            return self.get_best_next_action(actions, ground_truth) # 根据历史动作集actions选择最优动作

    def select_action_model(self, state):
        """
            测试过程--一律选择最佳动作
            模型根据状态选择动作
        """
        with torch.no_grad():
                if use_cuda:
                    inpu = Variable(state).cuda()
                else:
                    inpu = Variable(state)
                qval = self.policy_net(inpu)    # 各动作打分
                _, predicted = torch.max(qval.data,1)   # 最优动作的索引; 1表示取行最值，返回最值和索引
                #print("Predicted : "+str(qval.data))
                action = predicted[0]   # 取出最佳的动作
                #print(action)
                return action

    def optimize_model(self):
        """
        执行网络更新步骤的功能（批量采样、损耗计算、反向传播）
        """
        # 内存过小不执行更新
        # 尚未优化
        if len(self.memory) < self.BATCH_SIZE:  # 批次大小为100(执行100次再进行更新参数的操作)
            return

        # 提取随机样本,并构成对象
        # 随机选择, 避免连续存在的偏差
        transitions = self.memory.sample(self.BATCH_SIZE)   # 从缓存中选出一个批度数据(不连续)
        batch = Transition(*zip(*transitions))  # batch = collections.namedtuple('Batch', [state, int(action), next_state, reward])
        
        # 分离对象中的元素
        non_final_mask = torch.Tensor(tuple(map(lambda s: s is not None, batch.next_state))).bool() # torch.Size([100]); tensor类型,100个值,是否为None; 将batch.next_state(100个)通过lambda表达式,list->tuple->tensor;
        next_states = [s for s in batch.next_state if s is not None]
        # non_final_next_states = Variable(torch.cat(next_states), volatile=True).type(Tensor)
        with torch.no_grad():
            non_final_next_states = Variable(torch.cat(next_states)).type(Tensor)   # 拼接全部的next_state并转换为tensor

        state_batch = Variable(torch.cat(batch.state)).type(Tensor)     # torch.Size([100, 25169]); 拼接batch.state,并转换为tensor
        if use_cuda:
            state_batch = state_batch.cuda()
        action_batch = Variable(torch.LongTensor(batch.action).view(-1,1)).type(LongTensor)     # torch.Size([100, 1]), 选择的动作索引[0,8]
        # print("action_batch", action_batch)
        reward_batch = Variable(torch.FloatTensor(batch.reward).view(-1,1)).type(Tensor)


        # 通过Q网络传递状态计算动作打分, 并检索选定的操作(gether->根据action_batch替换action位置索引的列索引, 再根据位置索引找出动作集中的值)--预测值
        # 举例: action_batch = [[2,1,0]]; 位置索引=[(0,0), (0,1), (0,2)]; 替换后的位置索引=[(0,2), (0,1), (0,0)]; 然后提取指定位置的值
        state_action_values = self.policy_net(state_batch).gather(1, action_batch)  # torch.Size([100, 1]); 获取对动作的打分
        # 计算下一个状态的v（s{t+1}）。
        next_state_values = Variable(torch.zeros(self.BATCH_SIZE, 1).type(Tensor))  # 1列batch_size行, 初始值为0(如果是结束状态值就为0不变)

        if use_cuda:    # 需要传入的话就传到cuda
            non_final_next_states = non_final_next_states.cuda()
        
        # 调用第二个Q网络（复制以确保学习的稳定性）
        d = self.target_net(non_final_next_states)  # torch.Size([..., 9])
        next_state_values[non_final_mask] = d.max(1)[0].view(-1,1)  # max(1)取出每个动作组中的最值和下标,[0]取最值,然后放到一列
        # next_state_values.volatile = False

        # 计算预期Q函数值（使用奖励）--近似真实值
        expected_state_action_values = (next_state_values * self.GAMMA) + reward_batch

        # 计算损失函数(交叉熵损失)
        loss = criterion(state_action_values, expected_state_action_values)

        # 反向传播(策略网络)
        self.optimizer.zero_grad()  # 梯度清零
        loss.backward() # 反向传播
        self.optimizer.step()   # 更新参数
        
    
    def compose_state(self, image, dtype=FloatTensor):
        """
            状态组合 = 图片特征向量 + 历史动作集
        """
        image_feature = self.get_features(image, dtype)     # 获取从图片中提取得到的特征(维度[1, 512, 7, 7])
        # print("origin image feature: " + str(image_feature.shape))
        image_feature = image_feature.view(1,-1)    # 修改维度为1行(维度[1, 25088])
        # print("image feature : " + str(image_feature.shape))
        history_flatten = self.actions_history.view(1,-1).type(dtype)   # 动作空间类型
        # print("action history: ", self.actions_history)
        # print("action history flatten's shape: " + str(history_flatten.shape))
        state = torch.cat((image_feature, history_flatten), 1)  # 按列拼接(左右拼接), shape=[1, 25169]
        return state
    
    def get_features(self, image, dtype=FloatTensor):
        """
            从图像中提取特征向量
        """
        global transform
        #image = transform(image)
        # print("image shape: ", image.shape)
        # print("*image shape: ", *image.shape)
        image = image.view(1, *image.shape)     # 使用*将image.shape中的数据直接读出(image.shape是使用张量存储的数据); 使用view重构,(1,3,224,224)
        # print("new image shape:", image.shape)
        image = Variable(image).type(dtype)     # 转换成Variable类型, 自动求梯度
        if use_cuda:    # 用不用cuda
            image = image.cuda()
        feature = self.feature_extractor(image) # 提取图片特征
        #print("Feature shape : "+str(feature.shape))
        return feature.data

    
    def update_history(self, action):
        """
            添加新的动作到历史动作集(共保存9个动作组, 每个动作组都有9种动作,只有一个1)
            参数 :
                - 动作索引
        """
        action_vector = torch.zeros(9)
        action_vector[action] = 1   # 动作向量组, 选择的动作为1, 其余动作为0
        size_history_vector = len(torch.nonzero(self.actions_history))  # 历史动作集中非0动作的个数
        if size_history_vector < 9: # 不足9个动作 -> 将动作添加到最后一个动作中(最后一个动作集中动作action设为1)
            self.actions_history[size_history_vector][action] = 1
        else:   # 超过9个动作 -> 将[0,7]共8个动作后移, 将当前动作保存到第一个(0)位置
            for i in range(8,0,-1):
                self.actions_history[i][:] = self.actions_history[i-1][:]
            self.actions_history[0][:] = action_vector[:] 
        return self.actions_history

    def calculate_position_box(self, actions, xmin=0, xmax=224, ymin=0, ymax=224):
        """
            执行所有操作，生成标注框的最终坐标
        """
        # 计算修改尺度
        alpha_h = self.alpha * (  ymax - ymin )
        alpha_w = self.alpha * (  xmax - xmin )
        real_x_min, real_x_max, real_y_min, real_y_max = 0, 224, 0, 224

        # 根据动作修改标注框位置坐标
        for r in actions:
            if r == 1: # Right
                real_x_min += alpha_w
                real_x_max += alpha_w
            if r == 2: # Left
                real_x_min -= alpha_w
                real_x_max -= alpha_w
            if r == 3: # Up 
                real_y_min -= alpha_h
                real_y_max -= alpha_h
            if r == 4: # Down
                real_y_min += alpha_h
                real_y_max += alpha_h
            if r == 5: # Bigger
                real_y_min -= alpha_h
                real_y_max += alpha_h
                real_x_min -= alpha_w
                real_x_max += alpha_w
            if r == 6: # Smaller
                real_y_min += alpha_h
                real_y_max -= alpha_h
                real_x_min += alpha_w
                real_x_max -= alpha_w
            if r == 7: # Fatter
                real_y_min += alpha_h
                real_y_max -= alpha_h
            if r == 8: # Taller
                real_x_min += alpha_w
                real_x_max -= alpha_w
        real_x_min, real_x_max, real_y_min, real_y_max = self.rewrap(real_x_min), self.rewrap(real_x_max), self.rewrap(real_y_min), self.rewrap(real_y_max) # 选出[0,224]范围内的最小坐标
        return [real_x_min, real_x_max, real_y_min, real_y_max]

    def get_max_bdbox(self, ground_truth_boxes, actual_coordinates ):
        """
            获取最佳标注框坐标
            参数 :
                - ground_truth_boxes 真实标注框(图片中可能有多个同类内容)
                - actual_coordinates 构造的标注框
        """
        max_iou = False
        max_gt = []
        for gt in ground_truth_boxes:   # 遍历图片中的标注框
            iou = self.intersection_over_union(actual_coordinates, gt)  # 计算我们的标注框和真实标注框的交并比
            if max_iou == False or max_iou < iou:   # 找出最大的交并比max_iouh和边界框坐标max_gt
                max_iou = iou
                max_gt = gt
        return max_gt

    def predict_image(self, image, plot=False):
        """
            预测图像的标注框
        """

        # Q-Network转换到评估模式(dropout停止)
        self.policy_net.eval()
        xmin = 0
        xmax = 224
        ymin = 0
        ymax = 224

        done = False
        all_actions = []
        self.actions_history = torch.ones((9,9))    # 9*9 全1
        state = self.compose_state(image)   # 状态组合 = 图片特征向量 + 历史动作集
        original_image = image.clone()
        new_image = image

        steps = 0
        
        # 执行40步 或 选择动作0
        while not done:
            steps += 1  # 执行动作数
            action = self.select_action_model(state)
            all_actions.append(action)      # 收集选择的动作
            if action == 0:     # 停止动作
                next_state = None
                new_equivalent_coord = self.calculate_position_box(all_actions)     # 执行所有的动作获得标注框坐标
                done = True
            else:   # 动作[1,8], 是一个可执行动作
                self.actions_history = self.update_history(action)  # 更新历史动作集
                new_equivalent_coord = self.calculate_position_box(all_actions)     # 执行所有的动作获得标注框坐标
                
                new_image = original_image[:, int(new_equivalent_coord[2]):int(new_equivalent_coord[3]), int(new_equivalent_coord[0]):int(new_equivalent_coord[1])]     # 标注框框住的图片(先y后x)
                try:
                    new_image = transform(new_image)    # 处理图片为3*224*224
                except ValueError:
                    break            
                
                # 成分：状态+过去9个动作的历史记录
                next_state = self.compose_state(new_image)      # 状态组合 = 图片特征向量 + 历史动作集
            
            if steps == 40:
                done = True
            
            # 移动到新状态, 更新图片为新图片(和真实标注框的差距更小)
            state = next_state
            image = new_image
        
            if plot:
                show_new_bdbox(original_image, new_equivalent_coord, color='b', count=steps)
        

        # Génération d'un GIF représentant l'évolution de la prédiction
        if plot:
            #images = []
            tested = 0
            while os.path.isfile('media/movie_'+str(tested)+'.gif'):
                tested += 1
            # filepaths
            fp_out = "media/movie_"+str(tested)+".gif"
            images = []
            for count in range(1, steps+1):
                images.append(imageio.imread(str(count)+".png"))
            
            imageio.mimsave(fp_out, images)
            
            for count in range(1, steps):
                os.remove(str(count)+".png")
        return new_equivalent_coord


    
    def evaluate(self, dataset):
        """
            在数据集上评估模型性能
            入口 :
                - 测试数据集
            出口 :
                - 召回率和平均精确度DataFrame

        """
        ground_truth_boxes = []
        predicted_boxes = []
        print("Predicting boxes...")
        for key, value in dataset.items():   # 图片文件名 图片信息
            image, gt_boxes = extract(key, dataset)     # 图片 标注框
            bbox = self.predict_image(image)    # 预测图像的标注框
            ground_truth_boxes.append(gt_boxes)     # 收集真实标注框
            predicted_boxes.append(bbox)    # 收集预测标注框
        print("Computing recall and ap...")
        stats = eval_stats_at_threshold(predicted_boxes, ground_truth_boxes)    # DataFrame; 平均精度AP和召回率recall
        print("Final result : \n"+str(stats))   # 输出指标
        return stats

    def train(self, train_loader):
        """
            训练数据集
        """
        # 初始的边框是整张图片
        xmin = 0.0
        xmax = 224.0
        ymin = 0.0
        ymax = 224.0

        for i_episode in range(self.num_episodes):  # 每个类别训练15次 生成一个类别的网络并保存
            print("Episode "+str(i_episode))
            # if os.path.exists(self.save_path+"_"+self.classe):    # 加载训练好的模型

            for key, value in  train_loader.items():    # 图片文件名 图片信息
                image, ground_truth_boxes = extract(key, train_loader)  # 图片 标注框
                original_image = image.clone()      # 保留一张原始图片; 完全拷贝, 保存image的所有信息到original_image
                # print("origin image shape:" + str(original_image.shape))
                ground_truth = ground_truth_boxes[0]    # 图片中的第一个标注框
                all_actions = []
        
                # 初始化环境和状态空间
                self.actions_history = torch.ones((9,9))
                state = self.compose_state(image)   # 状态 = 图片特征向量 + 动作历史
                original_coordinates = [xmin, xmax, ymin, ymax]     # 初始的标注框
                new_image = image
                done = False    # 一回合结束标志
                t = 0
                actual_equivalent_coord = original_coordinates  # 实际标注框初始化
                new_equivalent_coord = original_coordinates     # 新标注框初始化
                while not done:     # 结束一张图片的标志：执行20次动作 或 action=0
                    t += 1
                    action = self.select_action(state, all_actions, ground_truth)   # 选择一个动作
                    all_actions.append(action)      # 收集选择的动作
                    if action == 0: # 结束标志
                        next_state = None
                        new_equivalent_coord = self.calculate_position_box(all_actions)     # 执行所有的动作获得标注框坐标
                        closest_gt = self.get_max_bdbox( ground_truth_boxes, new_equivalent_coord ) # 最佳标注框
                        reward = self.compute_trigger_reward(new_equivalent_coord,  closest_gt) # 新标注框和最佳标注框的差距; 不低于阈值->3, 低于阈值->(-3)
                        done = True # 一回合结束

                    else:
                        self.actions_history = self.update_history(action)  # 更新历史动作集
                        new_equivalent_coord = self.calculate_position_box(all_actions) # 根据历史动作集计算标注框坐标
                        
                        new_image = original_image[:, int(new_equivalent_coord[2]):int(new_equivalent_coord[3]), int(new_equivalent_coord[0]):int(new_equivalent_coord[1])]     # 标注框框住的图片(先y后x)
                        # print("new image before transform" + str(new_image.shape))
                        try:
                            new_image = transform(new_image)    # 调整图片尺寸为(3,224,224)
                        except ValueError:
                            break
                        # print("new image after transform" + str(new_image.shape))
                        next_state = self.compose_state(new_image)  # 状态组合 = 图片特征向量 + 动作历史
                        closest_gt = self.get_max_bdbox( ground_truth_boxes, new_equivalent_coord ) # 最佳标注框坐标
                        reward = self.compute_reward(new_equivalent_coord, actual_equivalent_coord, closest_gt) # 新旧两个标注框和最佳标注框的差距 计算奖励
                        
                        actual_equivalent_coord = new_equivalent_coord  # 更新标注框
                    if t == 20:
                        done = True
                    self.memory.push(state, int(action), next_state, reward)    # 更新缓存

                    # 更新 状态 图片(缩小图片范围)
                    state = next_state
                    image = new_image
                    # 网络更新(策略网络)
                    self.optimize_model()
                    
            
            if i_episode % self.TARGET_UPDATE == 0:     # 训练完一张图片,使用策略网络的参数更新目标网络
                self.target_net.load_state_dict(self.policy_net.state_dict())
            
            if i_episode<5:     # 前5次EPS递减
                self.EPS -= 0.18
            self.save_network()

            print('Complete')   # 完成一张图片的训练

    def train_validate(self, train_loader, valid_loader):
        """
            使用模型测试数据,将结果保存到日志文件中
        """
        op = open("logs_over_epochs", "w")
        op.write("NU = "+str(self.nu))
        op.write("ALPHA = "+str(self.alpha))
        op.write("THRESHOLD = "+str(self.threshold))
        xmin = 0.0
        xmax = 224.0
        ymin = 0.0
        ymax = 224.0
        for i_episode in range(self.num_episodes):  
            print("Episode "+str(i_episode))
            for key, value in  train_loader.items():
                image, ground_truth_boxes = extract(key, train_loader)
                original_image = image.clone()
                ground_truth = ground_truth_boxes[0]
                all_actions = []
        
                # Initialize the environment and state
                self.actions_history = torch.ones((9,9))
                state = self.compose_state(image)
                original_coordinates = [xmin, xmax, ymin, ymax]
                new_image = image
                done = False
                t = 0
                actual_equivalent_coord = original_coordinates
                new_equivalent_coord = original_coordinates
                while not done:
                    t += 1
                    action = self.select_action(state, all_actions, ground_truth)
                    all_actions.append(action)
                    if action == 0:
                        next_state = None
                        new_equivalent_coord = self.calculate_position_box(all_actions)
                        closest_gt = self.get_max_bdbox( ground_truth_boxes, new_equivalent_coord )
                        reward = self.compute_trigger_reward(new_equivalent_coord,  closest_gt)
                        done = True

                    else:
                        self.actions_history = self.update_history(action)
                        new_equivalent_coord = self.calculate_position_box(all_actions)
                        
                        new_image = original_image[:, int(new_equivalent_coord[2]):int(new_equivalent_coord[3]), int(new_equivalent_coord[0]):int(new_equivalent_coord[1])]
                        try:
                            new_image = transform(new_image)
                        except ValueError:
                            break                        
                        if False:
                            show_new_bdbox(original_image, ground_truth, color='r')
                            show_new_bdbox(original_image, new_equivalent_coord, color='b')
                            

                        next_state = self.compose_state(new_image)
                        closest_gt = self.get_max_bdbox( ground_truth_boxes, new_equivalent_coord )
                        reward = self.compute_reward(new_equivalent_coord, actual_equivalent_coord, closest_gt)
                        
                        actual_equivalent_coord = new_equivalent_coord
                    if t == 20:
                        done = True
                    self.memory.push(state, int(action), next_state, reward)

                    # Vers le nouvel état
                    state = next_state
                    image = new_image
                    # Optimisation
                    self.optimize_model()
                    
            stats = self.evaluate(valid_loader)     # DataFrame; 平均精度和召回率
            op.write("\n")  # 文件中换行
            op.write("Episode "+str(i_episode))     # 写入迭代次数
            op.write(str(stats))    # 写入评估指标
            if i_episode % self.TARGET_UPDATE == 0: # 每次使用策略网络参数更新目标网络参数
                self.target_net.load_state_dict(self.policy_net.state_dict())
            
            if i_episode<5:     # 前5次,EPS值递减; epsilon值, 提高最佳动作的选择概率
                self.EPS -= 0.18
            self.save_network()     # 保存网络
            
            print('Complete')