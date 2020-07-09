import os
from functools import reduce

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

# from data import voc, coco
from data import voc
from layers import *

class SKConv(nn.Module):
    def __init__(self,in_channels,out_channels,stride=1,M=2,r=16,L=32):
        '''
        :param in_channels:  输入通道维度
        :param out_channels: 输出通道维度   原论文中 输入输出通道维度相同
        :param stride:  步长，默认为1
        :param M:  分支数
        :param r: 特征Z的长度，计算其维度d 时所需的比率（论文中 特征S->Z 是降维，故需要规定 降维的下界）
        :param L:  论文中规定特征Z的下界，默认为32
        '''
        super(SKConv,self).__init__()
        d=max(in_channels//r,L)   # 计算向量Z 的长度d
        self.M=M
        self.out_channels=out_channels
        self.conv=nn.ModuleList()  # 根据分支数量 添加 不同核的卷积操作
        for i in range(M):
            # 为提高效率，原论文中 扩张卷积5x5为 （3X3，dilation=2）来代替。 且论文中建议组卷积G=32
            self.conv.append(nn.Sequential(nn.Conv2d(in_channels,out_channels,3,stride,padding=1+i,dilation=1+i,groups=1,bias=False),
                                           nn.BatchNorm2d(out_channels),
                                           nn.ReLU(inplace=True)))
        self.global_pool=nn.AdaptiveAvgPool2d(1) # 自适应pool到指定维度    这里指定为1，实现 GAP
        self.fc1=nn.Sequential(nn.Conv2d(out_channels,d,1,bias=False),
                               nn.BatchNorm2d(d),
                               nn.ReLU(inplace=True))   # 降维
        self.fc2=nn.Conv2d(d,out_channels*M,1,1,bias=False)  # 升维
        self.softmax=nn.Softmax(dim=1) # 指定dim=1  使得两个全连接层对应位置进行softmax,保证 对应位置a+b+..=1
    def forward(self, input):
        batch_size=input.size(0)
        output=[]
        #the part of split
        for i,conv in enumerate(self.conv):
            #print(i,conv(input).size())
            output.append(conv(input))
        #the part of fusion
        U=reduce(lambda x,y:x+y,output) # 逐元素相加生成 混合特征U
        s=self.global_pool(U)
        z=self.fc1(s)  # S->Z降维
        a_b=self.fc2(z) # Z->a，b 升维  论文使用conv 1x1表示全连接。结果中前一半通道值为a,后一半为b
        a_b=a_b.reshape(batch_size,self.M,self.out_channels,-1) #调整形状，变为 两个全连接层的值
        a_b=self.softmax(a_b) # 使得两个全连接层对应位置进行softmax
        #the part of selection
        a_b=list(a_b.chunk(self.M,dim=1))#split to a and b   chunk为pytorch方法，将tensor按照指定维度切分成 几个tensor块
        a_b=list(map(lambda x:x.reshape(batch_size,self.out_channels,1,1),a_b)) # 将所有分块  调整形状，即扩展两维
        V=list(map(lambda x,y:x*y,output,a_b)) # 权重与对应  不同卷积核输出的U 逐元素相乘
        V=reduce(lambda x,y:x+y,V) # 两个加权后的特征 逐元素相加
        return V

class SSD(nn.Module):
    """Single Shot Multibox Architecture
    The network is composed of a base VGG network followed by the
    added multibox conv layers.  Each multibox layer branches into
        1) conv2d for class conf scores
        2) conv2d for localization predictions
        3) associated priorbox layer to produce default bounding
           boxes specific to the layer's feature map size.
    See: https://arxiv.org/pdf/1512.02325.pdf for more details.
    Args:
        phase: (string) Can be "test" or "train"
        size: input image size
        base: VGG16 layers for input, size of either 300 or 500
        extras: extra layers that feed to multibox loc and conf layers
        head: "multibox head" consists of loc and conf conv layers
    """

    def __init__(self, phase, size, base, num_classes):
        super(SSD, self).__init__()
        self.phase = phase
        self.num_classes = num_classes
        # self.cfg = (coco, voc)[num_classes == 21]
        self.cfg = voc
        self.priorbox = PriorBox(self.cfg)
        # handbook
        # self.priors = Variable(self.priorbox.forward(), volatile=True)
        self.priors = self.priorbox.forward()
        # handbook
        self.size = size

        loc_layers = []
        conf_layers = []

        # SSD network
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')
        self.max_pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout = nn.Dropout2d()

        self.sk_conv_1 = SKConv(in_channels=3, out_channels=64, M=2)
        multibox_loc_1 = nn.Conv2d(192, 4*4, kernel_size=3, padding=1)
        multibox_conf_1 = nn.Conv2d(128+256, 4*2, kernel_size=7, padding=3)
        loc_layers.append(multibox_loc_1)
        conf_layers.append(multibox_conf_1)

        self.sk_conv_2 = SKConv(in_channels=64, out_channels=128, M=2)
        multibox_loc_2 = nn.Conv2d(384, 4*4, kernel_size=3, padding=1)
        multibox_conf_2 = nn.Conv2d(256+512, 4*2, kernel_size=7, padding=3)
        loc_layers.append(multibox_loc_2)
        conf_layers.append(multibox_conf_2)

        self.sk_conv_3 = SKConv(in_channels=128, out_channels=256, M=2)
        multibox_loc_3 = nn.Conv2d(768, 4*4, kernel_size=3, padding=1)
        multibox_conf_3 = nn.Conv2d(512+512, 4*2, kernel_size=7, padding=3)
        loc_layers.append(multibox_loc_3)
        conf_layers.append(multibox_conf_3)

        self.sk_conv_4 = SKConv(in_channels=256, out_channels=512, M=2)
        multibox_loc_4 = nn.Conv2d(1024, 4*4, kernel_size=3, padding=1)
        multibox_conf_4 = nn.Conv2d(512, 4*2, kernel_size=7, padding=3)
        loc_layers.append(multibox_loc_4)
        conf_layers.append(multibox_conf_4)

        self.sk_conv_5 = SKConv(in_channels=512, out_channels=512, M=2)
        # multibox_loc_5 = nn.Conv2d(512, 4*4, kernel_size=3, padding=1)
        # multibox_conf_5 = nn.Conv2d(512, 4*2, kernel_size=3, padding=1)
        # loc_layers.append(multibox_loc_5)
        # conf_layers.append(multibox_conf_5)

        self.loc = nn.ModuleList(loc_layers)
        self.conf = nn.ModuleList(conf_layers)

        # demo実行時
        if phase == 'test':
            self.softmax = nn.Softmax(dim=-1)
            # PyTorch1.5.0 support new-style autograd function
            #self.detect = Detect(num_classes, 0, 200, 0.01, 0.45)
            self.detect = Detect()
            # PyTorch1.5.0 support new-style autograd function

    # 順伝播
    def forward(self, x):
        """Applies network layers and ops on input image(s) x.
        Args:
            x: input image or batch of images. Shape: [batch,3,300,300].
        Return:
            Depending on phase:
            test:
                Variable(tensor) of output class label predictions,
                confidence score, and corresponding location predictions for
                each object detected. Shape: [batch,topk,7]
            train:
                list of concat outputs from:
                    1: confidence layers, Shape: [batch*num_priors,num_classes]
                    2: localization layers, Shape: [batch,num_priors*4]
                    3: priorbox layers, Shape: [2,num_priors*4]
        """
        sources = list()
        loc = list()
        conf = list()

        x = self.sk_conv_1(x)
        x = self.max_pool(x)
        feature_map_1 = x

        x = self.sk_conv_2(x)
        x = self.max_pool(x)
        feature_map_2 = x

        x = self.sk_conv_3(x)
        x = self.max_pool(x)
        feature_map_3 = x

        x = self.sk_conv_4(x)
        x = self.max_pool(x)
        feature_map_4 = x

        x = self.sk_conv_5(x)
        x = self.max_pool(x)
        feature_map_5 = x

        fpn_map_loc_1 = torch.cat((feature_map_1, self.upsample(feature_map_2)), 1)
        fpn_map_loc_2 = torch.cat((feature_map_2, self.upsample(feature_map_3)), 1)
        fpn_map_loc_3 = torch.cat((feature_map_3, self.upsample(feature_map_4)), 1)
        fpn_map_loc_4 = torch.cat((feature_map_4, self.upsample(feature_map_5)), 1)

        fpn_map_conf_1 = torch.cat((feature_map_2, self.upsample(feature_map_3)), 1)
        fpn_map_conf_2 = torch.cat((feature_map_3, self.upsample(feature_map_4)), 1)
        fpn_map_conf_3 = torch.cat((feature_map_4, self.upsample(feature_map_5)), 1)
        fpn_map_conf_4 = feature_map_5

        sources_loc = []
        sources_loc.append(fpn_map_loc_1)
        sources_loc.append(fpn_map_loc_2)
        sources_loc.append(fpn_map_loc_3)
        sources_loc.append(fpn_map_loc_4)

        sources_conf = []
        sources_conf.append(fpn_map_conf_1)
        sources_conf.append(fpn_map_conf_2)
        sources_conf.append(fpn_map_conf_3)
        sources_conf.append(fpn_map_conf_4)

        # apply multibox head to source layers
        for (x, l) in zip(sources_loc, self.loc):
            # (バッチサイズ,C,W,H) → (バッチサイズ,W,H,C)にTranspose
            x = l(x)
            loc.append(x.permute(0, 2, 3, 1).contiguous())

        for (x, c) in zip(sources_conf, self.conf):
            # (バッチサイズ,C,W,H) → (バッチサイズ,W,H,C)にTranspose
            # x = self.dropout(x)
            x = c(x)
            x = self.upsample(x)
            conf.append(x.permute(0, 2, 3, 1).contiguous())

        loc = torch.cat([o.view(o.size(0), -1) for o in loc], 1)
        conf = torch.cat([o.view(o.size(0), -1) for o in conf], 1)
        # demo実行時
        if self.phase == "test":
            # PyTorch1.5.0 support new-style autograd function
            #output = self.detect(
            output = self.detect.apply(self.num_classes, 0, 200, 0.01, 0.45,
            # PyTorch1.5.0 support new-style autograd function
                loc.view(loc.size(0), -1, 4),                   # loc preds
                self.softmax(conf.view(conf.size(0), -1,
                             self.num_classes)),                # conf preds
                self.priors.type(type(x.data))                  # default boxes
            )
        else:
        # train実行時
            output = (
                loc.view(loc.size(0), -1, 4),
                conf.view(conf.size(0), -1, self.num_classes),
                self.priors
            )
        return output

    def load_weights(self, base_file):
        other, ext = os.path.splitext(base_file)
        if ext == '.pkl' or '.pth':
            print('Loading weights into state dict...')
            self.load_state_dict(torch.load(base_file,
                                 map_location=lambda storage, loc: storage))
            print('Finished!')
        else:
            print('Sorry only .pth and .pkl files supported.')

# 特徴マップ毎のアスペクト比の数
mbox = {
    '512': [6, 6, 4, 4],
}

# ネットワークのリスト作成
def build_ssd(phase, size=512, num_classes=2):
    if phase != "test" and phase != "train":
        print("ERROR: Phase: " + phase + " not recognized")
        return

    return SSD(phase, size, mbox[str(size)], num_classes)
