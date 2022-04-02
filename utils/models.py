import torch.nn as nn
import torchvision


"""
    使用VGG16的第一部分用于特征提取
"""
class FeatureExtractor(nn.Module):
    def __init__(self):
        super(FeatureExtractor, self).__init__()
        vgg16 = torchvision.models.vgg16(pretrained=True)   # pretrained会自动下载权重并加载到模型中
        vgg16.eval() # to not do dropout
        self.features = list(vgg16.children())[0]   # vgg16分为features、avgpool、classifier三部分，这里取第一部分
        self.classifier = nn.Sequential(*list(vgg16.classifier.children())[:-2])    # 取classifier中[0:-2)的部分
    def forward(self, x):
        x = self.features(x)
        return x
    
"""
    文章所属Q-Network结构
"""
class DQN(nn.Module):
    def __init__(self, h, w, outputs):
        super(DQN, self).__init__()
        self.classifier = nn.Sequential(
            nn.Linear( in_features= 81 + 25088, out_features=1024),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear( in_features= 1024, out_features=1024),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear( in_features= 1024, out_features=9)
        )
    def forward(self, x):
        return self.classifier(x)