from collections import OrderedDict

from typing import Dict

import cv2
import numpy as np
import torch
from torch import nn, Tensor
from torch.autograd import Variable
from torch.nn import functional as F
from models.backbone import resnet50, resnet101


class IntermediateLayerGetter(nn.ModuleDict):

    _version = 2
    __annotations__ = {
        "return_layers": Dict[str, str],
    }

    def __init__(self, model: nn.Module, return_layers: Dict[str, str]) -> None:
        #判断key值是否在骨干网络中 'layer4': 'out'
        if not set(return_layers).issubset([name for name, _ in model.named_children()]):
            raise ValueError("return_layers are not present in model")
        #记录保存之前的return_layers
        orig_return_layers = return_layers
        #强制转化为str
        return_layers = {str(k): str(v) for k, v in return_layers.items()}

        # 重新构建backbone，将没有使用到的模块全部删掉
        #将之前我们需要的backbone部分，也就是到layer4 部分的网络保存下来，直到return_layers为空
        layers = OrderedDict()
        for name, module in model.named_children():
            layers[name] = module
            if name in return_layers:
                del return_layers[name]
            if not return_layers:
                break
        #调用父类初始化方法重构
        super(IntermediateLayerGetter, self).__init__(layers)
        self.return_layers = orig_return_layers

    def forward(self, x: Tensor) -> Dict[str, Tensor]:
        out = OrderedDict()
        for name, module in self.items():
            x = module(x)
            #return_layers = {'layer4': 'out','layer3' = 'aux'}
            #out_name = aux
            # out_name=out
            if name in self.return_layers:
                out_name = self.return_layers[name]
                # out[aux] = layer3(x)
                # out[out] = layer4(x)
                out[out_name] = x
        return out


class FCN(nn.Module):

    __constants__ = ['aux_classifier']

    def __init__(self, backbone, classifier, aux_classifier=None):
        super(FCN, self).__init__()
        self.backbone = backbone
        self.classifier = classifier
        self.aux_classifier = aux_classifier

    def forward(self, x: Tensor) -> Dict[str, Tensor]:
        input_shape = x.shape[-2:]
        # contract: features is a dict of tensors
        features = self.backbone(x)
        #初始化返回，有序字典
        result = OrderedDict()
        #将out的输出也就是layer4的输出输入到分类器中 FCN32
        x = features["out"]
        x = self.classifier(x)
        # 原论文中虽然使用的是ConvTranspose2d，但权重是冻结的，所以就是一个bilinear插值
        x = F.interpolate(x, size=input_shape, mode='bilinear', align_corners=False)
        result["out"] = x
        #返回辅助分类器的值
        if self.aux_classifier is not None:
            x = features["aux"]
            x = self.aux_classifier(x)
            # 原论文中虽然使用的是ConvTranspose2d，但权重是冻结的，所以就是一个bilinear插值
            x = F.interpolate(x, size=input_shape, mode='bilinear', align_corners=False)
            result["aux"] = x
        #result:{'out':[480,480,21],'aux':[480,480,21]}
        return result


class FCNHead(nn.Sequential):
    def __init__(self, in_channels, channels):
        '''
        1024->1024/4->21
        :param in_channels: 输入通道数l3 1024  或l4 2048
        :param channels:21
        '''
        inter_channels = in_channels // 4
        layers = [
            nn.Conv2d(in_channels, inter_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Conv2d(inter_channels, channels, 1)
        ]

        super(FCNHead, self).__init__(*layers)


def fcn_resnet50(aux, num_classes=21, pretrain_backbone=False):
    # 'resnet50_imagenet': 'https://download.pytorch.org/models/resnet50-0676ba61.pth'
    # 'fcn_resnet50_coco': 'https://download.pytorch.org/models/fcn_resnet50_coco-1167a1af.pth'
    backbone = resnet50(replace_stride_with_dilation=[False, True, True])

    if pretrain_backbone:
        # 载入resnet50 backbone预训练权重
        pthfile = r'D:\Program Files (x86)\models\resnet50-0676ba61.pth'
        backbone.load_state_dict(torch.load(pthfile, map_location='cpu'))
    #最终layer4通道数
    out_inplanes = 2048
    #辅助layer3通道数
    aux_inplanes = 1024

    return_layers = {'layer4': 'out'}
    if aux:
        return_layers['layer3'] = 'aux'
    #重构bcakbone
    backbone = IntermediateLayerGetter(backbone, return_layers=return_layers)
    #是否使用辅助分类器
    aux_classifier = None
    # why using aux: https://github.com/pytorch/vision/issues/4292
    #构建辅助分类器
    if aux:
        aux_classifier = FCNHead(aux_inplanes, num_classes)

    classifier = FCNHead(out_inplanes, num_classes)
    #构建FCN
    model = FCN(backbone, classifier, aux_classifier)

    return model


def fcn_resnet101(aux, num_classes=21, pretrain_backbone=False):
    # 'resnet101_imagenet': 'https://download.pytorch.org/models/resnet101-63fe2227.pth'
    # 'fcn_resnet101_coco': 'https://download.pytorch.org/models/fcn_resnet101_coco-7ecb50ca.pth'
    backbone = resnet101(replace_stride_with_dilation=[False, True, True])

    if pretrain_backbone:
        # 载入resnet101 backbone预训练权重
        backbone.load_state_dict(torch.load("resnet101.pth", map_location='cpu'))

    out_inplanes = 2048
    aux_inplanes = 1024

    return_layers = {'layer4': 'out'}
    if aux:
        return_layers['layer3'] = 'aux'
    backbone = IntermediateLayerGetter(backbone, return_layers=return_layers)

    aux_classifier = None
    # why using aux: https://github.com/pytorch/vision/issues/4292
    if aux:
        aux_classifier = FCNHead(aux_inplanes, num_classes)

    classifier = FCNHead(out_inplanes, num_classes)

    model = FCN(backbone, classifier, aux_classifier)

    return model


def get_test_input():
    img = cv2.imread("train.jpg")
    img = cv2.resize(img, (480,480))          #Resize to the input dimension
    img_ =  img[:,:,::-1].transpose((2,0,1))  # BGR -> RGB | H X W C -> C X H X W
    img_ = img_[np.newaxis,:,:,:]/255.0       #Add a channel at 0 (for batch) | Normalise
    img_ = torch.from_numpy(img_).float()     #Convert to float
    img_ = Variable(img_)                     # Convert to Variable
    return img_


def create_model(aux, num_classes, pretrain=True):
    #创建fcn_50
    model = fcn_resnet50(aux=aux, num_classes=num_classes)
    #加载预训练模型
    if pretrain:
        path = r"D:\Program Files (x86)\models\fcn_resnet50_coco-1167a1af.pth"
        weights_dict = torch.load(path, map_location='cpu')
        if num_classes != 21:
            # 官方提供的预训练权重是21类(包括背景)
            # 如果训练自己的数据集，将和类别相关的权重删除，防止权重shape不一致报错
            for k in list(weights_dict.keys()):
                if "classifier.4" in k:
                    del weights_dict[k]

        missing_keys, unexpected_keys = model.load_state_dict(weights_dict, strict=False)
        if len(missing_keys) != 0 or len(unexpected_keys) != 0:
            print("missing_keys: ", missing_keys)
            print("unexpected_keys: ", unexpected_keys)

    return model

if __name__ == '__main__':

    #model = fcn_resnet50(aux=True, num_classes=20)
    model = create_model(aux=True,num_classes=20,pretrain=True)
    print(model)
    inp = get_test_input()
    pred = model(inp)

    print(pred['out'].shape)
    print(pred['aux'].shape)
    #torch.Size([1, 20, 480, 480])
    #torch.Size([1, 20, 480, 480])