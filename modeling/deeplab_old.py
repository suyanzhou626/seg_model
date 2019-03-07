import torch
import torch.nn as nn
import torch.nn.functional as F
from .nn.aspp_old import build_aspp
from .nn.decoder import build_decoder
from .nn.xception_old import AlignedXception

class DeepLab(nn.Module):
    def __init__(self, args,backbone='resnet', output_stride=16, freeze_bn=False):
        super(DeepLab, self).__init__()
        num_classes = args.num_classes
        BatchNorm = args.batchnorm_function

        if backbone == 'drn':
            output_stride = 8

        self.backbone = AlignedXception(output_stride, BatchNorm)
        self.aspp = build_aspp(backbone, output_stride, BatchNorm)
        self.decoder = build_decoder(num_classes, backbone, BatchNorm)

        if freeze_bn:
            self.freeze_bn()

    def forward(self, input):
        x, low_level_feat = self.backbone(input)
        x = self.aspp(x)
        x = self.decoder(x, low_level_feat)
        x = F.interpolate(x, size=input.size()[2:], mode='bilinear', align_corners=True)

        return x

    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()

    def get_conv_weight_params(self):
        for m in self.named_modules():
            if isinstance(m[1],nn.Conv2d):
                for p in m[1].named_parameters():
                    if p[1].requires_grad:
                        if p[0] == 'weight':
                            yield p[1]

    def get_conv_bias_params(self):
        for m in self.named_modules():
            if isinstance(m[1],nn.Conv2d):
                for p in m[1].named_parameters():
                    if p[1].requires_grad:
                        if p[0] == 'bias':
                            yield p[1]

    def get_bn_prelu_params(self):
        for m in self.named_parameters():
            if 'bn' in m[0] or 'relu' in m[0]:
                if m[1].requires_grad:
                    yield m[1]


if __name__ == "__main__":
    import argparse
    import torch
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    args.num_classes = 2
    args.batchnorm_function = torch.nn.BatchNorm2d
    model = DeepLab(args)
    tempa = []
    tempb = []
    for i in model.named_parameters():
        tempa.append(i[0])

    for m in model.named_modules():
        if isinstance(m[1],torch.nn.Conv2d):
            for p in m[1].named_parameters():
                if p[1].requires_grad:
                    tempb.append(m[0]+'.'+p[0])
    for m in model.named_parameters():
        if 'bn' in m[0] or 'relu' in m[0]:
            if m[1].requires_grad:
                tempb.append(m[0])
    tempa = sorted(tempa)
    tempb = sorted(tempb)
    print(tempa == tempb)
    for i in tempa:
        if i not in tempb:
            print(i)
    num = 0
    for i in model.parameters():
        num += 1
    num1 = 0
    for i in model.get_conv_weight_params():
        num1 += 1
    for i in model.get_conv_bias_params():
        num1 += 1
    for i in model.get_bn_prelu_params():
        num1 += 1
    print(num,num1)
    model.eval()
    input = torch.rand(1, 3, 225,225)
    output = model(input)
    print(output.size())

