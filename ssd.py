import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from layers import *
# from data import voc, coco
from data import voc
import os

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

        self.conv_1_1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.conv_1_2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.max_pool_1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv_1 = nn.Conv2d(64, 128, kernel_size=1, padding=0)
        multibox_loc_1 = nn.Conv2d(128, 4*4,kernel_size=3,padding=1)
        multibox_conf_1 = nn.Conv2d(128, 4*2,kernel_size=3,padding=1)
        loc_layers.append(multibox_loc_1)
        conf_layers.append(multibox_conf_1)

        self.conv_2_1 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv_2_2 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.max_pool_2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv_2 = nn.Conv2d(128, 256, kernel_size=1, padding=0)
        multibox_loc_2 = nn.Conv2d(256, 4*4,kernel_size=3,padding=1)
        multibox_conf_2 = nn.Conv2d(256, 4*2,kernel_size=3,padding=1)
        loc_layers.append(multibox_loc_2)
        conf_layers.append(multibox_conf_2)

        self.conv_3_1 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.conv_3_2 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.conv_3_3 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.max_pool_3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv_3 = nn.Conv2d(256, 512, kernel_size=1, padding=0)
        multibox_loc_3 = nn.Conv2d(512, 4*4,kernel_size=3,padding=1)
        multibox_conf_3 = nn.Conv2d(512, 4*2,kernel_size=3,padding=1)
        loc_layers.append(multibox_loc_3)
        conf_layers.append(multibox_conf_3)

        self.conv_4_1 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.conv_4_2 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv_4_3 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.max_pool_4 = nn.MaxPool2d(kernel_size=2, stride=2)
        multibox_loc_4 = nn.Conv2d(512, 4*4,kernel_size=3,padding=1)
        multibox_conf_4 = nn.Conv2d(512, 4*2,kernel_size=3,padding=1)
        loc_layers.append(multibox_loc_4)
        conf_layers.append(multibox_conf_4)

        self.conv_5_1 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv_5_2 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv_5_3 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.max_pool_5 = nn.MaxPool2d(kernel_size=2, stride=2)
        multibox_loc_5 = nn.Conv2d(512, 4*4,kernel_size=3,padding=1)
        multibox_conf_5 = nn.Conv2d(512, 4*2,kernel_size=3,padding=1)
        loc_layers.append(multibox_loc_5)
        conf_layers.append(multibox_conf_5)

        # self.vgg = nn.ModuleList(base)
        # Layer learns to scale the l2 normalized features from conv4_3
        # self.L2Norm = L2Norm(512, 20)
        # self.extras = nn.ModuleList(extras)
        # オフセットと確信度のネットワークリスト
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

        x = self.conv_1_1(x)
        x = self.conv_1_2(x)
        x = self.max_pool_1(x)
        x = F.relu(x, inplace=True)
        feature_map_1 = x

        x = self.conv_2_1(x)
        x = self.conv_2_2(x)
        x = self.max_pool_2(x)
        x = F.relu(x, inplace=True)
        feature_map_2 = x

        x = self.conv_3_1(x)
        x = self.conv_3_2(x)
        x = self.conv_3_3(x)
        x = self.max_pool_3(x)
        x = F.relu(x, inplace=True)
        feature_map_3 = x

        x = self.conv_4_1(x)
        x = self.conv_4_2(x)
        x = self.conv_4_3(x)
        x = self.max_pool_4(x)
        x = F.relu(x, inplace=True)
        feature_map_4 = x

        x = self.conv_5_1(x)
        x = self.conv_5_2(x)
        x = self.conv_5_3(x)
        x = self.max_pool_5(x)
        x = F.relu(x, inplace=True)
        feature_map_5 = x

        fpn_map_1 = self.conv_1(feature_map_1) + self.upsample(feature_map_2)
        fpn_map_2 = self.conv_2(feature_map_2) + self.upsample(feature_map_3)
        fpn_map_3 = self.conv_3(feature_map_3) + self.upsample(feature_map_4)
        fpn_map_4 = feature_map_4 + self.upsample(feature_map_5)

        sources.append(fpn_map_1)
        sources.append(fpn_map_2)
        sources.append(fpn_map_3)
        sources.append(fpn_map_4)
        sources.append(feature_map_5)

        # apply multibox head to source layers
        for (x, l, c) in zip(sources, self.loc, self.conf):
            # (バッチサイズ,C,W,H) → (バッチサイズ,W,H,C)にTranspose
            loc.append(l(x).permute(0, 2, 3, 1).contiguous())
            conf.append(c(x).permute(0, 2, 3, 1).contiguous())

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
    '512': [2, 2, 2, 2, 2],
}

# ネットワークのリスト作成
def build_ssd(phase, size=512, num_classes=2):
    if phase != "test" and phase != "train":
        print("ERROR: Phase: " + phase + " not recognized")
        return

    return SSD(phase, size, mbox[str(size)], num_classes)
