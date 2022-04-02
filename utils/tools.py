import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import pandas as pd
from config import *
import random

classes = ['cat', 'bird', 'motorbike', 'diningtable', 'train', 'tvmonitor', 'bus', 'horse', 'car', 'pottedplant',
           'person', 'chair', 'boat', 'bottle', 'bicycle', 'dog', 'aeroplane', 'cow', 'sheep', 'sofa']


def sort_class_extract(datasets):
    """
        整理数据, 返回一个数据集字典
        {类别1: {图片文件名: 图片信息org}, ...}
    """
    datasets_per_class = {}  # {'cat': {}, 'bird': {}, 'motorbike': {}, ..., 'sofa': {}}, 类别字典，字典值也是一个字典
    for j in classes:
        datasets_per_class[j] = {}

    for dataset in datasets:  # 取出每个训练集的数据(这里是2007和2012年的训练数据 和 2007和2012年的测试数据)
        for i in dataset:  # 遍历数据集的每条数据(两个年份的数据(训练/测试))
            img, target = i  # 取出图片(tensor)和目标(dict)
            classe = target['annotation']['object'][0]["name"]  # 获取图片所属的第一个类别(可能一张图片中包含多个类别)
            filename = target['annotation']['filename']  # 图片的文件名

            org = {}
            for j in classes:  # {'cat': [img], 'bird': [img], 'motorbike': [img], ..., 'sofa': [img]}， 类别-图片字典，每一类装着当前的图片
                org[j] = []
                org[j].append(img)
            for i in range(len(target['annotation']['object'])):
                # 在上一步的基础上添加; 当前图片中存在类别的信息(可能多个; 保存两个信息)
                # {'cat': [img, [标注框, 图片尺寸]]}
                classe = target['annotation']['object'][i]["name"]  # 图片中包含的各个类别内容的名称
                org[classe].append(
                    [target['annotation']['object'][i]["bndbox"], target['annotation']['size']])  # 添加标注框 和 图片尺寸

            for j in classes:
                if len(org[j]) > 1:  # 类别j存在于当前图片中(不存在的话只保存有图片img,即长度为1)
                    try:
                        # 在datasets_per_class保存当前图片和信息org; 举例: {'cat': {'图片文件名': org}}
                        datasets_per_class[j][filename].append(org[j])
                    except KeyError:
                        # 还没有出现过这个文件 -> 先在这个文件名创建一个空列表,再填充信息; 举例: {'cat': {'图片文件名': [org]}}
                        datasets_per_class[j][filename] = []
                        datasets_per_class[j][filename].append(org[j])
    return datasets_per_class


def show_new_bdbox(image, labels, color='r', count=0):
    """
        查看标注框对应的图片
    """
    xmin, xmax, ymin, ymax = labels[0], labels[1], labels[2], labels[3]
    fig, ax = plt.subplots(1)
    ax.imshow(image.transpose(0, 2).transpose(0, 1))

    width = xmax - xmin
    height = ymax - ymin
    rect = patches.Rectangle((xmin, ymin), width, height, linewidth=3, edgecolor=color, facecolor='none')
    ax.add_patch(rect)
    ax.set_title("Iteration " + str(count))
    plt.savefig(str(count) + '.png', dpi=100)


def extract(index, loader):
    """
        根据修改后的图片尺寸修正标注框坐标
        返回图片和标注框列表
    """
    extracted = loader[index]   # 根据文件名index获取图片信息, 包括图片、标注框和图片尺寸
    ground_truth_boxes = []     # 边界框坐标集，存放每张图片中的每个类别的标注框坐标
    for ex in extracted:
        img = ex[0]     # 图片
        bndbox = ex[1][0]   # 标注框
        size = ex[1][1]     # 图片尺寸
        # 根据指定的尺寸224修正标注框坐标
        xmin = (float(bndbox['xmin']) / float(size['width'])) * 224
        xmax = (float(bndbox['xmax']) / float(size['width'])) * 224

        ymin = (float(bndbox['ymin']) / float(size['height'])) * 224
        ymax = (float(bndbox['ymax']) / float(size['height'])) * 224

        ground_truth_boxes.append([xmin, xmax, ymin, ymax])
    return img, ground_truth_boxes


def voc_ap(rec, prec, voc2007=False):
    """
        计算PA和召回。如果VOC2007为真，则使用Pascal VOC2007论文中建议的测量值（11分法）
    """
    if voc2007:
        ap = 0.0
        for t in np.arange(0.0, 1.1, 0.1):
            if np.sum(rec >= t) == 0:
                p = 0
            else:
                p = np.max(prec[rec >= t])
            ap = ap + p / 11.0
    else:
        mrec = np.concatenate(([0.0], rec, [1.0]))
        mpre = np.concatenate(([0.0], prec, [0.0]))

        for i in range(mpre.size - 1, 0, -1):
            mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

        i = np.where(mrec[1:] != mrec[:-1])[0]
        ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap


def prec_rec_compute(bounding_boxes, gt_boxes, ovthresh):
    """
        根据地形真实性和预测之间的阈值，通过交叉口/联合进行精确计算和重新调用。
    """
    nd = len(bounding_boxes)    # 预测边界框个数
    npos = nd   # 预测边界框个数
    tp = np.zeros(nd)   # 预测标志框达标
    fp = np.zeros(nd)   # 预测标志框不达标
    d = 0   # 标记框计数

    for index in range(len(bounding_boxes)):
        box1 = bounding_boxes[index]
        box2 = gt_boxes[index][0]
        x11, x21, y11, y21 = box1[0], box1[1], box1[2], box1[3]
        x12, x22, y12, y22 = box2[0], box2[1], box2[2], box2[3]

        yi1 = max(y11, y12)
        xi1 = max(x11, x12)
        yi2 = min(y21, y22)
        xi2 = min(x21, x22)
        inter_area = max(((xi2 - xi1) * (yi2 - yi1)), 0)    # 交集
        #  并集 = 面积和 - 交集
        box1_area = (x21 - x11) * (y21 - y11)
        box2_area = (x22 - x12) * (y22 - y12)
        union_area = box1_area + box2_area - inter_area
        # 计算交并比
        iou = inter_area / union_area

        if iou > ovthresh:  # 交并比达标
            tp[d] = 1.0
        else:   # 交并比不达标
            fp[d] = 1.0
        d += 1

    # 精度和召回计算
    fp = np.cumsum(fp)  # 将数组按行累加
    tp = np.cumsum(tp)  # 将数组按行累加
    rec = tp / float(npos)  # 召回率
    prec = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)   # 精度

    return prec, rec


def compute_ap_and_recall(all_bdbox, all_gt, ovthresh):
    """
        计算VOC检测公用指标 -- 平均精度 召回率
    """
    prec, rec = prec_rec_compute(all_bdbox, all_gt, ovthresh)   # 精度 召回率
    ap = voc_ap(rec, prec, False)   # 平均精度
    return ap, rec[-1]


def eval_stats_at_threshold(all_bdbox, all_gt, thresholds=[0.1, 0.2, 0.3, 0.4, 0.5]):
    """
        评估和收集不同阈值的统计数据。
        all_bdbox 预测标注框
        all_gt 真实标注框
    """
    stats = {}
    for ovthresh in thresholds:
        ap, recall = compute_ap_and_recall(all_bdbox, all_gt, ovthresh) # 平均精度 召回率
        stats[ovthresh] = {'ap': ap, 'recall': recall}  # 根据阈值保存 平均精度AP和召回率recall
    stats_df = pd.DataFrame.from_records(stats) * 100   # 将字典转换为DataFrame
    return stats_df


"""
    保存中间过程用于训练
"""
class ReplayMemory(object):
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        # 随机取一个批次数据(不连续)
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)
